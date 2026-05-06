from __future__ import annotations

import threading
from pathlib import Path

from django.core.files import File
from django.db import transaction
from django.utils import timezone

from faceswap.models import FaceSwapJob
from faceswap.services.engine import EngineRouter, FaceSwapError


def process_job(job_id):
    job = FaceSwapJob.objects.get(id=job_id)
    job.status = FaceSwapJob.Status.PROCESSING
    job.started_at = timezone.now()
    job.error_message = ""
    job.save(update_fields=["status", "started_at", "error_message", "updated_at"])

    output_name = f"{job.id}.jpg"
    output_path = Path(job.source_image.storage.path(f"faceswap/output/{job.id}/{output_name}"))
    requested_provider = job.metadata.get("requested_provider")

    try:
        engine = EngineRouter().build(provider=requested_provider)
        result = engine.swap(
            Path(job.source_image.path),
            Path(job.target_image.path),
            output_path,
        )
    except FaceSwapError as exc:
        job.status = FaceSwapJob.Status.FAILED
        job.error_message = str(exc)
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "completed_at", "updated_at"])
        return
    except Exception:
        job.status = FaceSwapJob.Status.FAILED
        job.error_message = "Unexpected processing failure."
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "completed_at", "updated_at"])
        return

    with output_path.open("rb") as handle:
        job.output_image.save(output_name, File(handle), save=False)

    job.provider = result.provider
    job.metadata = result.metadata
    job.status = FaceSwapJob.Status.COMPLETED
    job.completed_at = timezone.now()
    job.save(
        update_fields=[
            "output_image",
            "provider",
            "metadata",
            "status",
            "completed_at",
            "updated_at",
        ]
    )


def enqueue_job(job_id, run_async=True):
    if not run_async:
        process_job(job_id)
        return

    def _start():
        thread = threading.Thread(target=process_job, args=(job_id,), daemon=True)
        thread.start()

    transaction.on_commit(_start)
