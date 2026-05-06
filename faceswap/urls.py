from django.urls import path

from . import views

app_name = "faceswap"

urlpatterns = [
    path("", views.FaceSwapHomeView.as_view(), name="home"),
    path("jobs/<uuid:job_id>/", views.FaceSwapJobDetailView.as_view(), name="job-detail"),
    path("api/faceswap/jobs/", views.FaceSwapJobCreateApiView.as_view(), name="api-job-create"),
    path("api/faceswap/jobs/<uuid:job_id>/", views.FaceSwapJobDetailApiView.as_view(), name="api-job-detail"),
]
