from pathlib import Path
import shutil
import uuid
from unittest.mock import patch

from django.test import SimpleTestCase
from PIL import Image

from faceswap.services.engine import DetectionResult, FaceSwapError, OpenCvFaceSwapEngine


class FaceSwapEngineTests(SimpleTestCase):
    def make_workspace_dir(self):
        path = Path("test-artifacts") / str(uuid.uuid4())
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_swap_creates_output_file(self):
        tmp_dir = self.make_workspace_dir()
        source = tmp_dir / "source.jpg"
        target = tmp_dir / "target.jpg"
        output = tmp_dir / "output.jpg"
        Image.new("RGB", (400, 400), "white").save(source)
        Image.new("RGB", (400, 400), "gray").save(target)

        engine = OpenCvFaceSwapEngine()

        with patch.object(
            engine,
            "detect_single_face",
            side_effect=[
                DetectionResult((50, 50, 120, 120), "opencv-haar", 1),
                DetectionResult((150, 140, 130, 130), "opencv-haar", 1),
            ],
        ):
            result = engine.swap(source, target, output)

        self.assertTrue(output.exists())
        self.assertEqual(result.provider, "opencv-haar->opencv-haar")

    def test_no_face_raises_error(self):
        tmp_dir = self.make_workspace_dir()
        image = tmp_dir / "image.jpg"
        Image.new("RGB", (400, 400), "white").save(image)
        engine = OpenCvFaceSwapEngine()
        engine.detector = type("Detector", (), {"detectMultiScale": lambda *args, **kwargs: []})()
        with self.assertRaises(FaceSwapError):
            engine.detect_single_face(image)
