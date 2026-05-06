"""
Offline InsightFace face swapper.

This script uses local InsightFace models and the InSwapper ONNX model to swap
one source face onto one target image. It does not call any external API or cloud
service. All inference runs locally through ONNX Runtime.

Requirements:
    pip install insightface onnxruntime opencv-python numpy

Optional GPU:
    pip install onnxruntime-gpu

Expected local model files:
    .insightface/models/buffalo_l/
    .insightface/models/inswapper_128.onnx

Example:
    python insightface_face_swapper.py source.jpg target.jpg output.jpg --device auto
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class FaceSwapError(Exception):
    """User-facing error for invalid input or model failure."""


@dataclass
class FaceSwapConfig:
    source_path: Path
    target_path: Path
    output_path: Path
    model_root: Path
    device: str
    det_size: tuple[int, int]
    source_face_index: int
    target_face_index: int
    enhance: bool
    show: bool


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FaceSwapError(f"Could not read image: {path}")
    return image


def choose_providers(device: str) -> tuple[list[str], int]:
    """
    Pick ONNX Runtime providers.

    ctx_id >= 0 tells InsightFace to use GPU context when CUDA is available.
    ctx_id = -1 uses CPU. The script falls back safely when CUDA is absent.
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"], -1

    if device in {"gpu", "cuda", "auto"} and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0

    if device in {"gpu", "cuda"}:
        raise FaceSwapError(
            "CUDAExecutionProvider is not available. Install onnxruntime-gpu or use --device cpu."
        )

    return ["CPUExecutionProvider"], -1


def load_models(model_root: Path, providers: list[str], ctx_id: int, det_size: tuple[int, int]):
    """Load local face analysis and swapper models."""
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    if not model_root.exists():
        raise FaceSwapError(f"Model root does not exist: {model_root}")

    swapper_path = model_root / "models" / "inswapper_128.onnx"
    if not swapper_path.exists():
        raise FaceSwapError(f"InSwapper model not found: {swapper_path}")

    app = FaceAnalysis(name="buffalo_l", root=str(model_root), providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    swapper = get_model(str(swapper_path), providers=providers)
    return app, swapper


def detect_faces(app, image: np.ndarray, label: str):
    """Detect faces and sort largest first for predictable default selection."""
    faces = app.get(image)
    if not faces:
        raise FaceSwapError(f"No face detected in {label} image.")

    faces = sorted(
        faces,
        key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
        reverse=True,
    )
    return faces


def select_face(faces, index: int, label: str):
    if index < 0 or index >= len(faces):
        raise FaceSwapError(f"{label} face index {index} is invalid. Detected {len(faces)} face(s).")
    return faces[index]


def enhance_result(image: np.ndarray) -> np.ndarray:
    """
    Lightweight offline enhancement for demo output.

    CLAHE improves local contrast in the luminance channel, and a mild bilateral
    filter softens small blending artifacts without destroying edges too much.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(enhanced, d=5, sigmaColor=25, sigmaSpace=25)


def draw_debug_boxes(image: np.ndarray, faces) -> np.ndarray:
    debug = image.copy()
    for index, face in enumerate(faces):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 220, 120), 2)
        cv2.putText(
            debug,
            str(index),
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 120),
            2,
            cv2.LINE_AA,
        )
    return debug


def swap_face(config: FaceSwapConfig) -> np.ndarray:
    providers, ctx_id = choose_providers(config.device)
    app, swapper = load_models(config.model_root, providers, ctx_id, config.det_size)

    source_image = read_image(config.source_path)
    target_image = read_image(config.target_path)

    source_faces = detect_faces(app, source_image, "source")
    target_faces = detect_faces(app, target_image, "target")

    source_face = select_face(source_faces, config.source_face_index, "Source")
    target_face = select_face(target_faces, config.target_face_index, "Target")

    result = swapper.get(target_image, target_face, source_face, paste_back=True)
    if result is None:
        raise FaceSwapError("InSwapper returned no output image.")

    if config.enhance:
        result = enhance_result(result)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(config.output_path), result)

    debug_dir = config.output_path.parent / f"{config.output_path.stem}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "source_faces.jpg"), draw_debug_boxes(source_image, source_faces))
    cv2.imwrite(str(debug_dir / "target_faces.jpg"), draw_debug_boxes(target_image, target_faces))

    print(f"Providers: {providers}")
    print(f"Source faces detected: {len(source_faces)}")
    print(f"Target faces detected: {len(target_faces)}")
    print(f"Saved result: {config.output_path}")
    print(f"Saved debug images: {debug_dir}")

    if config.show:
        cv2.imshow("Source", source_image)
        cv2.imshow("Target", target_image)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def parse_args() -> FaceSwapConfig:
    parser = argparse.ArgumentParser(description="Offline InsightFace/InSwapper face swapper.")
    parser.add_argument("source", help="Source image path: face to transfer.")
    parser.add_argument("target", help="Target image path: face to replace.")
    parser.add_argument("output", help="Output image path.")
    parser.add_argument("--model-root", default=".insightface", help="Local InsightFace model root.")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu", "cuda"], default="auto")
    parser.add_argument("--det-size", type=int, default=640, help="Detection size, e.g. 640.")
    parser.add_argument("--source-face-index", type=int, default=0, help="Which source face to use, largest face is 0.")
    parser.add_argument("--target-face-index", type=int, default=0, help="Which target face to replace, largest face is 0.")
    parser.add_argument("--no-enhance", action="store_true", help="Disable lightweight post-processing.")
    parser.add_argument("--show", action="store_true", help="Display source, target, and result windows.")
    args = parser.parse_args()

    return FaceSwapConfig(
        source_path=Path(args.source),
        target_path=Path(args.target),
        output_path=Path(args.output),
        model_root=Path(args.model_root).resolve(),
        device=args.device,
        det_size=(args.det_size, args.det_size),
        source_face_index=args.source_face_index,
        target_face_index=args.target_face_index,
        enhance=not args.no_enhance,
        show=args.show,
    )


def main() -> None:
    config = parse_args()
    try:
        swap_face(config)
    except FaceSwapError as exc:
        raise SystemExit(f"Face swap failed: {exc}") from exc


if __name__ == "__main__":
    main()
