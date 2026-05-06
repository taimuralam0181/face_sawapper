from pathlib import Path

from django import forms
from django.conf import settings


class FaceSwapJobForm(forms.Form):
    source_image = forms.ImageField()
    target_image = forms.ImageField()
    engine = forms.ChoiceField(choices=settings.FACESWAP_ALLOWED_ENGINES, initial="pretrained")

    def _validate_image(self, image, label):
        extension = Path(image.name).suffix.lower()
        if extension not in settings.FACESWAP_ALLOWED_EXTENSIONS:
            raise forms.ValidationError(
                f"{label} must be one of: {', '.join(sorted(settings.FACESWAP_ALLOWED_EXTENSIONS))}."
            )
        if image.size > settings.FACESWAP_MAX_UPLOAD_SIZE:
            raise forms.ValidationError(
                f"{label} exceeds {settings.FACESWAP_MAX_UPLOAD_SIZE // (1024 * 1024)}MB."
            )
        return image

    def clean_source_image(self):
        return self._validate_image(self.cleaned_data["source_image"], "Source image")

    def clean_target_image(self):
        return self._validate_image(self.cleaned_data["target_image"], "Target image")
