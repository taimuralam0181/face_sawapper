from django.contrib import admin

from .models import FaceSwapJob


@admin.register(FaceSwapJob)
class FaceSwapJobAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "status",
        "created_at",
        "updated_at",
        "provider",
        "has_output",
    )
    list_filter = ("status", "provider", "created_at")
    readonly_fields = (
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
        "source_image_preview",
        "target_image_preview",
        "output_image_preview",
    )
    search_fields = ("id", "error_message")

    @admin.display(boolean=True, description="Output")
    def has_output(self, obj):
        return bool(obj.output_image)

    def source_image_preview(self, obj):
        return obj.source_image.name if obj.source_image else "-"

    def target_image_preview(self, obj):
        return obj.target_image.name if obj.target_image else "-"

    def output_image_preview(self, obj):
        return obj.output_image.name if obj.output_image else "-"
