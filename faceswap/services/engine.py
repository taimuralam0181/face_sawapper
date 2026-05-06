from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from django.conf import settings


class FaceSwapError(Exception):
    """Structured error surfaced to the job runner."""


@dataclass
class SwapResult:
    output_path: Path
    provider: str
    metadata: dict


@dataclass
class DetectionResult:
    box: tuple[int, int, int, int]
    detector: str
    face_count: int


@dataclass
class FaceData:
    image: np.ndarray
    box: tuple[int, int, int, int]
    detector: str
    face_count: int
    keypoints: np.ndarray
    landmarks: np.ndarray


class OpenCvFaceSwapEngine:
    provider_name = "opencv-haar"

    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise FaceSwapError("OpenCV Haar cascade failed to load.")

    def _pick_primary_face(self, faces):
        if len(faces) == 0:
            return None
        return max(faces, key=lambda face: face[2] * face[3])

    def _detect_with_opencv(self, image_path: Path) -> DetectionResult | None:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FaceSwapError(f"Failed to read image: {image_path.name}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        primary = self._pick_primary_face(faces)
        if primary is None:
            return None
        x, y, w, h = primary
        return DetectionResult((int(x), int(y), int(w), int(h)), "opencv-haar", len(faces))

    def _detect_with_mtcnn(self, image_path: Path) -> DetectionResult | None:
        try:
            from mtcnn import MTCNN
        except Exception:
            return None

        image = cv2.imread(str(image_path))
        if image is None:
            raise FaceSwapError(f"Failed to read image: {image_path.name}")

        detector = MTCNN()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        if not faces:
            return None

        primary = max(faces, key=lambda face: face["box"][2] * face["box"][3])
        x, y, w, h = primary["box"]
        return DetectionResult((max(0, int(x)), max(0, int(y)), int(w), int(h)), "mtcnn", len(faces))

    def detect_single_face(self, image_path: Path) -> DetectionResult:
        errors = []
        for detector in (self._detect_with_opencv, self._detect_with_mtcnn):
            try:
                result = detector(image_path)
            except Exception as exc:
                errors.append(str(exc))
                continue
            if result is not None:
                return result
        detail = "; ".join(error for error in errors if error) or "No supported detector found a face."
        raise FaceSwapError(f"No face detected in {image_path.name}. {detail}")

    def swap(self, source_path: Path, target_path: Path, output_path: Path) -> SwapResult:
        src_box = self.detect_single_face(source_path)
        tgt_box = self.detect_single_face(target_path)
        source = Image.open(source_path).convert("RGBA")
        target = Image.open(target_path).convert("RGBA")

        sx, sy, sw, sh = src_box.box
        tx, ty, tw, th = tgt_box.box

        source_face = source.crop((sx, sy, sx + sw, sy + sh)).resize((tw, th), Image.Resampling.LANCZOS)
        mask = Image.new("L", (tw, th), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, tw, th), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(6, min(tw, th) // 10)))
        target.paste(source_face, (tx, ty), mask)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        target.convert("RGB").save(output_path, format="JPEG", quality=95)

        return SwapResult(
            output_path=output_path,
            provider=f"{src_box.detector}->{tgt_box.detector}",
            metadata={
                "source_face_box": src_box.box,
                "target_face_box": tgt_box.box,
                "source_detector": src_box.detector,
                "target_detector": tgt_box.detector,
                "source_face_count": src_box.face_count,
                "target_face_count": tgt_box.face_count,
                "pipeline": "fallback-paste",
            },
        )


class InsightFaceBase:
    def __init__(self) -> None:
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            raise FaceSwapError(f"InsightFace import failed: {exc}") from exc

        model_root = Path(settings.FACESWAP_MODEL_ROOT)
        model_root.mkdir(parents=True, exist_ok=True)
        providers = ["CPUExecutionProvider"]
        try:
            self.app = FaceAnalysis(name="buffalo_l", root=str(model_root), providers=providers)
            self.app.prepare(ctx_id=0, det_size=settings.FACESWAP_DETECTION_SIZE)
        except Exception as exc:
            raise FaceSwapError("Landmark model initialization failed. Check the local .insightface model cache.") from exc

    def _load_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FaceSwapError(f"Failed to read image: {image_path.name}")
        return image

    def _pick_primary_face(self, faces):
        if not faces:
            return None
        return max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))

    def _extract_face_data(self, image_path: Path) -> FaceData:
        image = self._load_image(image_path)
        faces = self.app.get(image)
        if not faces:
            raise FaceSwapError(f"No face detected in {image_path.name}.")
        face = self._pick_primary_face(faces)
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        landmarks = getattr(face, "landmark_2d_106", None)
        keypoints = getattr(face, "kps", None)
        if landmarks is None or keypoints is None:
            raise FaceSwapError("Required landmarks/keypoints are unavailable for local processing.")
        return FaceData(
            image=image,
            box=(x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
            detector="insightface",
            face_count=len(faces),
            keypoints=np.asarray(keypoints, dtype=np.float32),
            landmarks=np.asarray(landmarks, dtype=np.float32),
        )

    def _landmark_roi_brightness(self, face_data: FaceData) -> float:
        mask = np.zeros(face_data.image.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(np.int32(face_data.landmarks))
        cv2.fillConvexPoly(mask, hull, 255)
        gray = cv2.cvtColor(face_data.image, cv2.COLOR_BGR2GRAY)
        values = gray[mask > 0]
        return float(values.mean()) if values.size else float(gray.mean())

    def _pose_angle(self, face_data: FaceData) -> float:
        left_eye, right_eye = face_data.keypoints[0], face_data.keypoints[1]
        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])
        return float(np.degrees(np.arctan2(dy, dx)))

    def validate_pair(self, source_data: FaceData, target_data: FaceData, strict=True) -> dict:
        src_angle = self._pose_angle(source_data)
        tgt_angle = self._pose_angle(target_data)
        pose_diff = abs(src_angle - tgt_angle)

        src_brightness = self._landmark_roi_brightness(source_data)
        tgt_brightness = self._landmark_roi_brightness(target_data)
        brightness_diff = abs(src_brightness - tgt_brightness)

        src_area = float(source_data.box[2] * source_data.box[3])
        tgt_area = float(target_data.box[2] * target_data.box[3])
        face_ratio_diff = abs(np.log((src_area + 1e-6) / (tgt_area + 1e-6)))

        issues = []
        if source_data.face_count != 1:
            raise FaceSwapError("Source image must contain exactly one clear face.")
        if target_data.face_count != 1:
            raise FaceSwapError("Target image must contain exactly one clear face.")
        if pose_diff > settings.FACESWAP_MAX_POSE_ANGLE_DIFF:
            issues.append(f"pose mismatch {pose_diff:.1f} deg")
        if brightness_diff > settings.FACESWAP_MAX_BRIGHTNESS_DIFF:
            issues.append(f"lighting mismatch {brightness_diff:.1f}")
        if face_ratio_diff > settings.FACESWAP_MAX_FACE_RATIO_DIFF:
            issues.append("face size mismatch")
        if strict and issues:
            raise FaceSwapError(
                "Input validation failed: " + ", ".join(issues) + ". Use more similar source and target images."
            )

        return {
            "source_pose_angle": round(src_angle, 2),
            "target_pose_angle": round(tgt_angle, 2),
            "pose_diff": round(pose_diff, 2),
            "source_brightness": round(src_brightness, 2),
            "target_brightness": round(tgt_brightness, 2),
            "brightness_diff": round(brightness_diff, 2),
            "face_ratio_diff": round(float(face_ratio_diff), 4),
            "validation_issues": issues,
        }


class ClassicalInsightFaceSwapEngine(InsightFaceBase):
    provider_name = "classical-insightface"
    border_point_count = 8
    feather_ratio = 0.10
    jaw_feather_ratio = 0.18
    eye_suppression_ratio = 0.07
    mouth_suppression_ratio = 0.10

    def _add_border_points(self, image_shape, points: np.ndarray) -> np.ndarray:
        height, width = image_shape[:2]
        border = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1], [width // 2, 0], [width - 1, height // 2], [width // 2, height - 1], [0, height // 2]],
            dtype=np.float32,
        )
        return np.vstack([points, border])

    def _alignment_anchors(self, face_data: FaceData) -> np.ndarray:
        left_eye = face_data.keypoints[0]
        right_eye = face_data.keypoints[1]
        nose_tip = face_data.keypoints[2]
        mouth_left = face_data.keypoints[3]
        mouth_right = face_data.keypoints[4]
        mouth_center = (mouth_left + mouth_right) / 2.0
        return np.asarray([left_eye, right_eye, nose_tip, mouth_left, mouth_right, mouth_center], dtype=np.float32)

    def _normalize_face_region(self, source_data: FaceData, target_data: FaceData) -> FaceData:
        src_h, src_w = source_data.image.shape[:2]
        tx, ty, tw, th = target_data.box
        sx, sy, sw, sh = source_data.box
        target_scale = max(th / max(sh, 1), tw / max(sw, 1))
        cx = sx + sw / 2.0
        cy = sy + sh / 2.0
        crop_half_w = int(sw * target_scale * 1.2 / 2.0)
        crop_half_h = int(sh * target_scale * 1.2 / 2.0)
        x1 = max(0, int(cx - crop_half_w))
        y1 = max(0, int(cy - crop_half_h))
        x2 = min(src_w, int(cx + crop_half_w))
        y2 = min(src_h, int(cy + crop_half_h))
        crop = source_data.image[y1:y2, x1:x2]
        if crop.size == 0:
            return source_data
        resized = cv2.resize(crop, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
        scale_x = src_w / max(1, (x2 - x1))
        scale_y = src_h / max(1, (y2 - y1))
        keypoints = source_data.keypoints.copy()
        landmarks = source_data.landmarks.copy()
        keypoints[:, 0] = (keypoints[:, 0] - x1) * scale_x
        keypoints[:, 1] = (keypoints[:, 1] - y1) * scale_y
        landmarks[:, 0] = (landmarks[:, 0] - x1) * scale_x
        landmarks[:, 1] = (landmarks[:, 1] - y1) * scale_y
        return FaceData(resized, source_data.box, source_data.detector, source_data.face_count, keypoints, landmarks)

    def _align_source_to_target(self, source_data: FaceData, target_data: FaceData, output_shape) -> tuple[np.ndarray, np.ndarray]:
        source_data = self._normalize_face_region(source_data, target_data)
        matrix, _ = cv2.estimateAffinePartial2D(
            self._alignment_anchors(source_data),
            self._alignment_anchors(target_data),
            method=cv2.LMEDS,
        )
        if matrix is None:
            raise FaceSwapError("Face alignment failed.")
        width, height = output_shape[1], output_shape[0]
        aligned_image = cv2.warpAffine(source_data.image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        transformed_landmarks = cv2.transform(source_data.landmarks[None, :, :], matrix)[0]
        return aligned_image, transformed_landmarks

    def _triangle_indexes(self, points: np.ndarray, size) -> list[tuple[int, int, int]]:
        width, height = size
        subdiv = cv2.Subdiv2D((0, 0, width, height))
        for point in points:
            x = min(max(int(round(point[0])), 0), width - 1)
            y = min(max(int(round(point[1])), 0), height - 1)
            subdiv.insert((x, y))
        indexed, seen = [], set()
        for triangle in subdiv.getTriangleList():
            pts = [(triangle[0], triangle[1]), (triangle[2], triangle[3]), (triangle[4], triangle[5])]
            if any(x < 0 or y < 0 or x >= width or y >= height for x, y in pts):
                continue
            tri_idx = []
            for px, py in pts:
                distances = np.sum((points - np.array([px, py], dtype=np.float32)) ** 2, axis=1)
                tri_idx.append(int(np.argmin(distances)))
            if len(set(tri_idx)) != 3:
                continue
            key = tuple(sorted(tri_idx))
            if key in seen:
                continue
            seen.add(key)
            indexed.append(tuple(tri_idx))
        if not indexed:
            raise FaceSwapError("Triangulation failed; no valid face triangles were created.")
        return indexed

    def _warp_triangle(self, source_image, destination_canvas, source_triangle, target_triangle, accumulation_mask, weight_canvas) -> None:
        src_x, src_y, src_w, src_h = cv2.boundingRect(np.float32([source_triangle]))
        dst_x, dst_y, dst_w, dst_h = cv2.boundingRect(np.float32([target_triangle]))
        if min(src_w, src_h, dst_w, dst_h) <= 0:
            return
        src_crop = source_image[src_y: src_y + src_h, src_x: src_x + src_w]
        if src_crop.size == 0:
            return
        src_local = source_triangle - np.array([src_x, src_y], dtype=np.float32)
        dst_local = target_triangle - np.array([dst_x, dst_y], dtype=np.float32)
        matrix = cv2.getAffineTransform(src_local.astype(np.float32), dst_local.astype(np.float32))
        warped = cv2.warpAffine(src_crop, matrix, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((dst_h, dst_w), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(dst_local), 1.0, lineType=cv2.LINE_AA)
        mask = np.clip(cv2.GaussianBlur(mask, (7, 7), 0), 0.0, 1.0)
        mask_3 = mask[..., None].astype(np.float32)
        destination_region = destination_canvas[dst_y: dst_y + dst_h, dst_x: dst_x + dst_w]
        weight_region = weight_canvas[dst_y: dst_y + dst_h, dst_x: dst_x + dst_w]
        if destination_region.shape[:2] != warped.shape[:2]:
            return
        destination_region[:] = destination_region + warped.astype(np.float32) * mask_3
        weight_region[:] = weight_region + mask_3
        accumulation_mask[dst_y: dst_y + dst_h, dst_x: dst_x + dst_w] = np.maximum(accumulation_mask[dst_y: dst_y + dst_h, dst_x: dst_x + dst_w], mask.astype(np.float32))

    def _transfer_color(self, source_face: np.ndarray, target_face: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.max() <= 0:
            return np.clip(source_face, 0, 255).astype(np.uint8)
        source_bgr = np.clip(source_face, 0, 255).astype(np.uint8)
        target_bgr = np.clip(target_face, 0, 255).astype(np.uint8)
        source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        active = mask > 15
        if not np.any(active):
            return source_bgr
        corrected = source_lab.copy()
        for channel in range(3):
            src_vals = source_lab[:, :, channel][active]
            tgt_vals = target_lab[:, :, channel][active]
            src_mean, src_std = float(src_vals.mean()), float(src_vals.std() + 1e-6)
            tgt_mean, tgt_std = float(tgt_vals.mean()), float(tgt_vals.std() + 1e-6)
            channel_data = ((corrected[:, :, channel] - src_mean) * (tgt_std / src_std)) + tgt_mean
            corrected[:, :, channel][active] = channel_data[active]
        return cv2.cvtColor(np.clip(corrected, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _feather_mask(self, mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        if binary.max() == 0:
            return mask
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        feather_distance = max(6.0, float(min(mask.shape[:2])) * self.feather_ratio)
        feather = np.clip(distance / feather_distance, 0.0, 1.0)
        feather = cv2.GaussianBlur(feather.astype(np.float32), (21, 21), 0)
        return np.clip(feather * 255.0, 0, 255).astype(np.uint8)

    def _build_face_mask(self, landmarks: np.ndarray, image_shape) -> np.ndarray:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(np.int32(landmarks))
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask = self._feather_mask(mask)
        height, width = image_shape[:2]
        jaw_gradient = np.ones((height, width), dtype=np.float32)
        jaw_start = int(height * 0.58)
        if jaw_start < height:
            fade = np.linspace(1.0, 0.55, num=height - jaw_start, dtype=np.float32)
            jaw_gradient[jaw_start:, :] = fade[:, None]
        softened = np.clip(mask.astype(np.float32) * jaw_gradient, 0, 255).astype(np.uint8)
        return self._feather_mask(softened)

    def _build_feature_suppression_mask(self, face_data: FaceData, image_shape) -> np.ndarray:
        suppression = np.zeros(image_shape[:2], dtype=np.uint8)
        width, height = max(1, face_data.box[2]), max(1, face_data.box[3])
        eye_radius_x = max(6, int(width * self.eye_suppression_ratio))
        eye_radius_y = max(4, int(height * 0.045))
        mouth_radius_x = max(10, int(width * self.mouth_suppression_ratio))
        mouth_radius_y = max(6, int(height * 0.05))
        left_eye = tuple(np.int32(face_data.keypoints[0]))
        right_eye = tuple(np.int32(face_data.keypoints[1]))
        mouth_left = tuple(np.int32(face_data.keypoints[3]))
        mouth_right = tuple(np.int32(face_data.keypoints[4]))
        mouth_center = tuple(np.int32((face_data.keypoints[3] + face_data.keypoints[4]) / 2.0))
        cv2.ellipse(suppression, left_eye, (eye_radius_x, eye_radius_y), 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(suppression, right_eye, (eye_radius_x, eye_radius_y), 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(suppression, mouth_center, (mouth_radius_x, mouth_radius_y), 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)
        cv2.line(suppression, mouth_left, mouth_right, 255, thickness=max(3, mouth_radius_y // 2), lineType=cv2.LINE_AA)
        return cv2.GaussianBlur(suppression, (15, 15), 0)

    def swap(self, source_path: Path, target_path: Path, output_path: Path) -> SwapResult:
        source_data = self._extract_face_data(source_path)
        target_data = self._extract_face_data(target_path)
        preflight = self.validate_pair(source_data, target_data, strict=True)
        aligned_source_image, aligned_source_landmarks = self._align_source_to_target(source_data, target_data, target_data.image.shape)
        source_points = self._add_border_points(target_data.image.shape, aligned_source_landmarks)
        target_points = self._add_border_points(target_data.image.shape, target_data.landmarks)
        triangles = self._triangle_indexes(target_points, (target_data.image.shape[1], target_data.image.shape[0]))

        warped_face = np.zeros_like(target_data.image, dtype=np.float32)
        accumulation_mask = np.zeros(target_data.image.shape[:2], dtype=np.float32)
        weight_canvas = np.zeros_like(target_data.image, dtype=np.float32)
        for i, j, k in triangles:
            self._warp_triangle(
                aligned_source_image.astype(np.float32),
                warped_face,
                np.float32([source_points[i], source_points[j], source_points[k]]),
                np.float32([target_points[i], target_points[j], target_points[k]]),
                accumulation_mask,
                weight_canvas,
            )
        warped_face = np.clip(warped_face / np.maximum(weight_canvas, 1e-6), 0, 255).astype(np.uint8)
        face_mask = self._build_face_mask(target_data.landmarks, target_data.image.shape)
        face_mask = cv2.bitwise_and(face_mask, face_mask, mask=(accumulation_mask > 0.05).astype(np.uint8) * 255)
        face_mask = cv2.subtract(face_mask, self._build_feature_suppression_mask(target_data, target_data.image.shape))
        face_mask = self._feather_mask(face_mask)
        warped_face = self._transfer_color(warped_face.astype(np.float32), target_data.image.astype(np.float32), face_mask)
        mask_points = np.column_stack(np.where(face_mask > 0))
        if mask_points.size == 0:
            raise FaceSwapError("Generated face mask is empty; blending cannot continue.")
        center_y, center_x = np.mean(mask_points, axis=0).astype(int)
        blended = cv2.seamlessClone(warped_face, target_data.image, face_mask, (int(center_x), int(center_y)), cv2.MIXED_CLONE)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), blended)
        metadata = {
            "source_face_box": source_data.box,
            "target_face_box": target_data.box,
            "source_detector": source_data.detector,
            "target_detector": target_data.detector,
            "source_face_count": source_data.face_count,
            "target_face_count": target_data.face_count,
            "triangle_count": len(triangles),
            "landmark_count": int(target_data.landmarks.shape[0]),
            "feather_ratio": self.feather_ratio,
            "jaw_feather_ratio": self.jaw_feather_ratio,
            "eye_suppression_ratio": self.eye_suppression_ratio,
            "mouth_suppression_ratio": self.mouth_suppression_ratio,
            "pipeline": "anchor-alignment-triangulation-affine-color-transfer-seamless-clone",
        }
        metadata.update(preflight)
        return SwapResult(output_path=output_path, provider=self.provider_name, metadata=metadata)


class PretrainedInsightFaceSwapEngine(InsightFaceBase):
    provider_name = "pretrained-inswapper"

    def __init__(self) -> None:
        super().__init__()
        try:
            from insightface.model_zoo import get_model
        except Exception as exc:
            raise FaceSwapError(f"InsightFace swap model import failed: {exc}") from exc
        model_root = Path(settings.FACESWAP_MODEL_ROOT)
        providers = ["CPUExecutionProvider"]
        try:
            self.swapper = get_model(str(model_root / "models" / "inswapper_128.onnx"), providers=providers)
        except Exception as exc:
            raise FaceSwapError("Pretrained swap model initialization failed. Check .insightface/models/inswapper_128.onnx.") from exc

    def swap(self, source_path: Path, target_path: Path, output_path: Path) -> SwapResult:
        source_data = self._extract_face_data(source_path)
        target_data = self._extract_face_data(target_path)
        preflight = self.validate_pair(source_data, target_data, strict=False)

        source_face_obj = self._pick_primary_face(self.app.get(source_data.image))
        target_face_obj = self._pick_primary_face(self.app.get(target_data.image))
        swapped = self.swapper.get(target_data.image, target_face_obj, source_face_obj, paste_back=True)
        if swapped is None:
            raise FaceSwapError("Pretrained local model returned no output image.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), swapped)
        metadata = {
            "source_face_box": source_data.box,
            "target_face_box": target_data.box,
            "source_detector": source_data.detector,
            "target_detector": target_data.detector,
            "source_face_count": source_data.face_count,
            "target_face_count": target_data.face_count,
            "pipeline": "pretrained-local-inswapper",
        }
        metadata.update(preflight)
        return SwapResult(output_path=output_path, provider=self.provider_name, metadata=metadata)


class EngineRouter:
    def build(self, provider=None):
        provider = provider or settings.FACESWAP_MODEL_PROVIDER
        if provider == "pretrained":
            return PretrainedInsightFaceSwapEngine()
        if provider == "classical":
            return ClassicalInsightFaceSwapEngine()
        if provider == "opencv":
            return OpenCvFaceSwapEngine()
        raise FaceSwapError(f"Unsupported face swap provider: {provider}")
