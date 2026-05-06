import uuid

from django.db import models
from django.utils.deconstruct import deconstructible


@deconstructible
class UploadToPath:
    def __init__(self, prefix):
        self.prefix = prefix

    def __call__(self, instance, filename):
        return f"faceswap/{self.prefix}/{instance.id}/{filename}"


class FaceSwapJob(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    source_image = models.ImageField(upload_to=UploadToPath("source"))
    target_image = models.ImageField(upload_to=UploadToPath("target"))
    output_image = models.ImageField(
        upload_to=UploadToPath("output"),
        blank=True,
        null=True,
    )
    error_message = models.TextField(blank=True)
    provider = models.CharField(max_length=50, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"FaceSwapJob<{self.id}>"
