"""
Offline classical face swapper.

This script implements a thesis-friendly face swapping pipeline using only local
computer vision operations:

1. detect one face in source and target
2. extract landmarks
3. align source to target using stable anchors
4. triangulate target landmarks
5. affine-warp source triangles to target triangles
6. color-correct the warped face in LAB color space
7. build a feathered convex-hull face mask
8. blend once with cv2.seamlessClone

No external APIs or cloud services are used.

Example with dlib:
    python offline_face_swapper.py source.jpg target.jpg output.jpg --backend dlib --predictor shape_predictor_68_face_landmarks.dat

Example with MediaPipe:
    python offline_face_swapper.py source.jpg target.jpg output.jpg --backend mediapipe

Example with InsightFace landmarks already cached locally:
    python offline_face_swapper.py source.jpg target.jpg output.jpg --backend insightface
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FaceLandmarks:
    """Detected face data in image coordinates."""

    rect: tuple[int, int, int, int]
    landmarks: np.ndarray
    anchors: np.ndarray


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def detect_face_dlib(image: np.ndarray, predictor_path: str | Path) -> FaceLandmarks:
    """Detect exactly one face and 68 landmarks with dlib."""
    try:
        import dlib
    except ImportError as exc:
        raise RuntimeError("dlib is not installed. Install dlib or use --backend mediapipe.") from exc

    predictor_path = Path(predictor_path)
    if not predictor_path.exists():
        raise FileNotFoundError(f"dlib predictor not found: {predictor_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))
    faces = detector(gray, 1)

    if len(faces) != 1:
        raise ValueError(f"Expected exactly one face, found {len(faces)}.")

    face = faces[0]
    shape = predictor(gray, face)
    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)

    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)
    nose_tip = landmarks[30]
    mouth_left = landmarks[48]
    mouth_right = landmarks[54]
    anchors = np.array([left_eye, right_eye, nose_tip, mouth_left, mouth_right], dtype=np.float32)

    rect = (face.left(), face.top(), face.width(), face.height())
    return FaceLandmarks(rect=rect, landmarks=landmarks, anchors=anchors)


def detect_face_mediapipe(image: np.ndarray) -> FaceLandmarks:
    """Detect one face and dense landmarks with local MediaPipe Face Mesh."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe is not installed. Install mediapipe or use --backend dlib.") from exc
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "Installed mediapipe package does not expose mp.solutions.FaceMesh. "
            "Use --backend insightface in this environment, or install a MediaPipe version with FaceMesh solutions."
        )

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    result = face_mesh.process(rgb)
    face_mesh.close()

    if not result.multi_face_landmarks:
        raise ValueError("Expected exactly one face, found 0.")

    points = []
    for landmark in result.multi_face_landmarks[0].landmark:
        points.append((landmark.x * width, landmark.y * height))
    landmarks = np.array(points, dtype=np.float32)

    # MediaPipe Face Mesh anchor indices.
    # 33/133 and 362/263 are eye corners, 1 is nose tip, 61/291 are mouth corners.
    left_eye = (landmarks[33] + landmarks[133]) / 2.0
    right_eye = (landmarks[362] + landmarks[263]) / 2.0
    nose_tip = landmarks[1]
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]
    anchors = np.array([left_eye, right_eye, nose_tip, mouth_left, mouth_right], dtype=np.float32)

    x, y, w, h = cv2.boundingRect(landmarks.astype(np.int32))
    return FaceLandmarks(rect=(x, y, w, h), landmarks=landmarks, anchors=anchors)


def detect_face_insightface(image: np.ndarray) -> FaceLandmarks:
    """
    Detect one face using local InsightFace models.

    This is not a cloud/API dependency. It uses model files cached on disk and is
    useful when dlib or MediaPipe are not available for the active Python version.
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError("insightface is not installed. Use --backend dlib or --backend mediapipe instead.") from exc

    root = Path(".insightface").resolve()
    app = FaceAnalysis(name="buffalo_l", root=str(root), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(image)

    if len(faces) != 1:
        raise ValueError(f"Expected exactly one face, found {len(faces)}.")

    face = faces[0]
    landmarks = getattr(face, "landmark_2d_106", None)
    keypoints = getattr(face, "kps", None)
    if landmarks is None or keypoints is None:
        raise RuntimeError("InsightFace did not return dense landmarks/keypoints.")

    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    anchors = np.array(
        [
            keypoints[0],
            keypoints[1],
            keypoints[2],
            keypoints[3],
            keypoints[4],
        ],
        dtype=np.float32,
    )
    return FaceLandmarks(
        rect=(x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
        landmarks=np.asarray(landmarks, dtype=np.float32),
        anchors=anchors,
    )


def detect_face(image: np.ndarray, backend: str, predictor_path: str | None = None) -> FaceLandmarks:
    """Backend dispatcher for local face detection and landmark extraction."""
    if backend == "dlib":
        if not predictor_path:
            raise ValueError("--predictor is required when using --backend dlib.")
        return detect_face_dlib(image, predictor_path)
    if backend == "mediapipe":
        return detect_face_mediapipe(image)
    if backend == "insightface":
        return detect_face_insightface(image)
    raise ValueError(f"Unsupported backend: {backend}")


def align_face(
    source_image: np.ndarray,
    source_landmarks: np.ndarray,
    source_anchors: np.ndarray,
    target_anchors: np.ndarray,
    target_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align source face to target using stable anchor points before triangulation.

    estimateAffinePartial2D gives a similarity-like transform with translation,
    rotation, and scale while avoiding excessive shear.
    """
    matrix, _ = cv2.estimateAffinePartial2D(source_anchors, target_anchors, method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError("Could not estimate alignment transform.")

    target_h, target_w = target_shape[:2]
    aligned_image = cv2.warpAffine(
        source_image,
        matrix,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    aligned_landmarks = cv2.transform(source_landmarks[None, :, :], matrix)[0]
    return aligned_image, aligned_landmarks


def add_boundary_points(points: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    """Add image boundary points so Delaunay covers the full face canvas safely."""
    h, w = image_shape[:2]
    boundary = np.array(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
            [w // 2, 0],
            [w - 1, h // 2],
            [w // 2, h - 1],
            [0, h // 2],
        ],
        dtype=np.float32,
    )
    return np.vstack([points, boundary])


def triangulate(points: np.ndarray, image_shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """Run Delaunay triangulation over target landmarks and return triangle indices."""
    h, w = image_shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, w, h))

    for point in points:
        x = min(max(int(round(point[0])), 0), w - 1)
        y = min(max(int(round(point[1])), 0), h - 1)
        subdiv.insert((x, y))

    triangles = []
    seen = set()
    for triangle in subdiv.getTriangleList():
        tri_points = [(triangle[0], triangle[1]), (triangle[2], triangle[3]), (triangle[4], triangle[5])]
        if any(x < 0 or y < 0 or x >= w or y >= h for x, y in tri_points):
            continue

        indices = []
        for x, y in tri_points:
            distances = np.sum((points - np.array([x, y], dtype=np.float32)) ** 2, axis=1)
            indices.append(int(np.argmin(distances)))

        if len(set(indices)) != 3:
            continue
        key = tuple(sorted(indices))
        if key in seen:
            continue
        seen.add(key)
        triangles.append(tuple(indices))

    if not triangles:
        raise RuntimeError("Delaunay triangulation produced no valid triangles.")
    return triangles


def warp_one_triangle(
    source_image: np.ndarray,
    source_triangle: np.ndarray,
    target_triangle: np.ndarray,
    canvas: np.ndarray,
    weight_canvas: np.ndarray,
) -> None:
    """Warp one source triangle into target space using a soft anti-aliased mask."""
    src_rect = cv2.boundingRect(source_triangle.astype(np.float32))
    dst_rect = cv2.boundingRect(target_triangle.astype(np.float32))

    src_x, src_y, src_w, src_h = src_rect
    dst_x, dst_y, dst_w, dst_h = dst_rect
    if min(src_w, src_h, dst_w, dst_h) <= 0:
        return

    src_crop = source_image[src_y : src_y + src_h, src_x : src_x + src_w]
    if src_crop.size == 0:
        return

    src_local = source_triangle - np.array([src_x, src_y], dtype=np.float32)
    dst_local = target_triangle - np.array([dst_x, dst_y], dtype=np.float32)

    matrix = cv2.getAffineTransform(src_local.astype(np.float32), dst_local.astype(np.float32))
    warped = cv2.warpAffine(
        src_crop,
        matrix,
        (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((dst_h, dst_w), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_local), 1.0, lineType=cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = np.clip(mask, 0.0, 1.0)
    mask3 = mask[..., None]

    canvas_roi = canvas[dst_y : dst_y + dst_h, dst_x : dst_x + dst_w]
    weight_roi = weight_canvas[dst_y : dst_y + dst_h, dst_x : dst_x + dst_w]
    if canvas_roi.shape[:2] != warped.shape[:2]:
        return

    canvas_roi[:] += warped.astype(np.float32) * mask3
    weight_roi[:] += mask3


def warp_triangles(
    aligned_source: np.ndarray,
    aligned_source_points: np.ndarray,
    target_points: np.ndarray,
    triangle_indices: list[tuple[int, int, int]],
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Warp and blend all triangles into one floating-point face canvas."""
    canvas = np.zeros(target_shape, dtype=np.float32)
    weight_canvas = np.zeros(target_shape, dtype=np.float32)

    for i, j, k in triangle_indices:
        warp_one_triangle(
            aligned_source,
            np.float32([aligned_source_points[i], aligned_source_points[j], aligned_source_points[k]]),
            np.float32([target_points[i], target_points[j], target_points[k]]),
            canvas,
            weight_canvas,
        )

    normalized = canvas / np.maximum(weight_canvas, 1e-6)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def create_mask(landmarks: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    """Create a smooth convex-hull face mask from target landmarks."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(np.int32(landmarks))
    cv2.fillConvexPoly(mask, hull, 255)

    # Erode keeps hair/background outside the clone area.
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)

    # Blur and distance-transform feathering produce a softer jawline/edge.
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    binary = (mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    feather_radius = max(6.0, min(image_shape[:2]) * 0.08)
    feather = np.clip(distance / feather_radius, 0.0, 1.0)
    feather = cv2.GaussianBlur(feather.astype(np.float32), (21, 21), 0)
    return np.clip(feather * 255, 0, 255).astype(np.uint8)


def color_transfer_lab(source_face: np.ndarray, target_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match LAB channel mean/std from warped source face to target face region."""
    source_bgr = np.clip(source_face, 0, 255).astype(np.uint8)
    target_bgr = np.clip(target_image, 0, 255).astype(np.uint8)

    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    active = mask > 15
    if not np.any(active):
        return source_bgr

    corrected = source_lab.copy()
    for channel in range(3):
        src_values = source_lab[:, :, channel][active]
        tgt_values = target_lab[:, :, channel][active]

        src_mean = float(src_values.mean())
        src_std = float(src_values.std() + 1e-6)
        tgt_mean = float(tgt_values.mean())
        tgt_std = float(tgt_values.std() + 1e-6)

        adjusted = ((corrected[:, :, channel] - src_mean) * (tgt_std / src_std)) + tgt_mean
        corrected[:, :, channel][active] = adjusted[active]

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)


def blend_faces(
    corrected_face: np.ndarray,
    target_image: np.ndarray,
    mask: np.ndarray,
    clone_mode: str = "mixed",
) -> np.ndarray:
    """Blend the full warped face once with seamlessClone."""
    x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
    if w == 0 or h == 0:
        raise RuntimeError("Cannot blend because mask is empty.")

    center = (x + w // 2, y + h // 2)
    mode = cv2.MIXED_CLONE if clone_mode == "mixed" else cv2.NORMAL_CLONE
    return cv2.seamlessClone(corrected_face, target_image, mask, center, mode)


def debug_write(debug_dir: Path | None, name: str, image: np.ndarray) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / name), image)


def face_swap(
    source_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    backend: str = "dlib",
    predictor_path: str | None = None,
    clone_mode: str = "mixed",
    debug_dir: str | Path | None = None,
    show: bool = False,
) -> np.ndarray:
    """Complete offline face swapping pipeline."""
    debug_path = Path(debug_dir) if debug_dir else None

    source_image = read_image(source_path)
    target_image = read_image(target_path)

    source_face = detect_face(source_image, backend, predictor_path)
    target_face = detect_face(target_image, backend, predictor_path)

    aligned_source, aligned_source_landmarks = align_face(
        source_image,
        source_face.landmarks,
        source_face.anchors,
        target_face.anchors,
        target_image.shape,
    )
    debug_write(debug_path, "01_aligned_source.jpg", aligned_source)

    source_points = add_boundary_points(aligned_source_landmarks, target_image.shape)
    target_points = add_boundary_points(target_face.landmarks, target_image.shape)

    triangles = triangulate(target_points, target_image.shape)
    warped_face = warp_triangles(aligned_source, source_points, target_points, triangles, target_image.shape)
    debug_write(debug_path, "02_warped_face.jpg", warped_face)

    mask = create_mask(target_face.landmarks, target_image.shape)
    debug_write(debug_path, "03_mask.jpg", mask)

    corrected_face = color_transfer_lab(warped_face, target_image, mask)
    debug_write(debug_path, "04_color_corrected_face.jpg", corrected_face)

    result = blend_faces(corrected_face, target_image, mask, clone_mode=clone_mode)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)

    if show:
        cv2.imshow("Source", source_image)
        cv2.imshow("Target", target_image)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline classical face swapper.")
    parser.add_argument("source", help="Path to source image: face to transfer.")
    parser.add_argument("target", help="Path to target image: face to replace.")
    parser.add_argument("output", help="Path where result image will be saved.")
    parser.add_argument("--backend", choices=["dlib", "mediapipe", "insightface"], default="insightface")
    parser.add_argument("--predictor", help="Path to dlib shape_predictor_68_face_landmarks.dat.")
    parser.add_argument("--clone-mode", choices=["normal", "mixed"], default="mixed")
    parser.add_argument("--debug-dir", help="Optional directory for intermediate images.")
    parser.add_argument("--show", action="store_true", help="Display source, target, and result windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    face_swap(
        source_path=args.source,
        target_path=args.target,
        output_path=args.output,
        backend=args.backend,
        predictor_path=args.predictor,
        clone_mode=args.clone_mode,
        debug_dir=args.debug_dir,
        show=args.show,
    )
    print(f"Saved face swap result to: {args.output}")


if __name__ == "__main__":
    main()
