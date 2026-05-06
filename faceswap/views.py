from django.conf import settings
from django.http import Http404
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views import View
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .forms import FaceSwapJobForm
from .models import FaceSwapJob
from .services.job_runner import enqueue_job


def job_to_payload(job):
    return {
        "id": str(job.id),
        "status": job.status,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "result_image_url": job.output_image.url if job.output_image else None,
        "error": job.error_message or None,
        "provider": job.provider or None,
        "metadata": job.metadata,
    }


class FaceSwapHomeView(View):
    template_name = "faceswap/home.html"

    def get(self, request):
        form = FaceSwapJobForm()
        recent_jobs = FaceSwapJob.objects.all()[:5]
        return render(
            request,
            self.template_name,
            {"form": form, "recent_jobs": recent_jobs},
        )

    def post(self, request):
        form = FaceSwapJobForm(request.POST, request.FILES)
        recent_jobs = FaceSwapJob.objects.all()[:5]
        if not form.is_valid():
            return render(
                request,
                self.template_name,
                {"form": form, "recent_jobs": recent_jobs},
                status=400,
            )

        job = FaceSwapJob.objects.create(
            source_image=form.cleaned_data["source_image"],
            target_image=form.cleaned_data["target_image"],
            metadata={"requested_provider": form.cleaned_data["engine"]},
        )
        enqueue_job(job.id, run_async=settings.FACESWAP_RUN_ASYNC)
        job.refresh_from_db()
        return redirect(reverse("faceswap:job-detail", kwargs={"job_id": job.id}))


class FaceSwapJobDetailView(View):
    template_name = "faceswap/job_detail.html"

    def get_object(self, job_id):
        try:
            return FaceSwapJob.objects.get(id=job_id)
        except FaceSwapJob.DoesNotExist as exc:
            raise Http404 from exc

    def get(self, request, job_id):
        job = self.get_object(job_id)
        return render(request, self.template_name, {"job": job, "payload": job_to_payload(job)})


class FaceSwapJobCreateApiView(APIView):
    def post(self, request):
        form = FaceSwapJobForm(request.POST, request.FILES)
        if not form.is_valid():
            return Response({"errors": form.errors}, status=status.HTTP_400_BAD_REQUEST)

        job = FaceSwapJob.objects.create(
            source_image=form.cleaned_data["source_image"],
            target_image=form.cleaned_data["target_image"],
            metadata={"requested_provider": form.cleaned_data["engine"]},
        )
        enqueue_job(job.id, run_async=settings.FACESWAP_RUN_ASYNC)
        job.refresh_from_db()
        return Response(job_to_payload(job), status=status.HTTP_201_CREATED)


class FaceSwapJobDetailApiView(APIView):
    def get(self, request, job_id):
        try:
            job = FaceSwapJob.objects.get(id=job_id)
        except FaceSwapJob.DoesNotExist:
            return Response({"detail": "Job not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(job_to_payload(job))
