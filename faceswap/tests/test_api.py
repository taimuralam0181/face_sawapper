import io
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from PIL import Image

from faceswap.models import FaceSwapJob


def build_test_upload(name="image.jpg", color=(220, 220, 220)):
    image = Image.new("RGB", (256, 256), color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return SimpleUploadedFile(name, buffer.getvalue(), content_type="image/jpeg")


@override_settings(FACESWAP_RUN_ASYNC=False)
class FaceSwapApiTests(TestCase):
    def test_create_job_returns_completed_payload(self):
        def fake_process(job_id):
            job = FaceSwapJob.objects.get(id=job_id)
            output = SimpleUploadedFile("result.jpg", b"fake-result", content_type="image/jpeg")
            job.status = FaceSwapJob.Status.COMPLETED
            job.provider = "test-engine"
            job.output_image.save("result.jpg", output, save=False)
            job.metadata = {"mocked": True}
            job.save()

        with patch("faceswap.services.job_runner.process_job", side_effect=fake_process):
            response = self.client.post(
                "/api/faceswap/jobs/",
                {
                    "engine": "classical",
                    "source_image": build_test_upload("source.jpg"),
                    "target_image": build_test_upload("target.jpg", color=(100, 100, 100)),
                },
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["status"], FaceSwapJob.Status.COMPLETED)
        self.assertEqual(payload["provider"], "test-engine")
        self.assertIsNotNone(payload["result_image_url"])

    def test_invalid_extension_is_rejected(self):
        response = self.client.post(
            "/api/faceswap/jobs/",
            {
                "engine": "pretrained",
                "source_image": build_test_upload("source.gif"),
                "target_image": build_test_upload("target.jpg"),
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("errors", response.json())
