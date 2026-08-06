from django.urls import path
from eda_app import views

urlpatterns = [
    path("", views.index, name="index"),
    path("run/", views.run_pipeline, name="run_pipeline"),
    path("api/status/", views.pipeline_status, name="pipeline_status"),
    path("api/datasets/", views.sample_datasets, name="sample_datasets"),
    path("api/log/", views.pipeline_log, name="pipeline_log"),
    path("api/preview/", views.dataset_preview, name="dataset_preview"),
    path("reset/", views.reset_session, name="reset_session"),
    path("submit-answer/", views.submit_answer, name="submit_answer"),
    path("download-report/", views.download_report, name="download_report"),
    path("artifact/<path:artifact_path>", views.serve_artifact, name="serve_artifact"),
]
