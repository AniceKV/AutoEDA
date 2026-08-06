"""
Tests for eda_app Django views and API endpoints.
"""
import io
import json
import os
from django.test import TestCase, Client
from django.urls import reverse
from django.conf import settings
from eda_app import pipeline_runner


class DjangoAppViewsTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.session_id = "test-session-123"

    def tearDown(self):
        pipeline_runner.clear_state(self.session_id)

    def test_index_view(self):
        """Test main index page renders correctly."""
        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "eda_app/index.html")
        self.assertIn("sample_csvs", response.context)
        self.assertIn("overview", response.context)

    def test_sample_datasets_api(self):
        """Test API listing available sample CSV datasets."""
        response = self.client.get(reverse("sample_datasets"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("files", data)
        self.assertIsInstance(data["files"], list)

    def test_pipeline_status_idle(self):
        """Test initial status polling endpoint returns idle."""
        response = self.client.get(reverse("pipeline_status"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "idle")
        self.assertFalse(data["done"])

    def test_pipeline_log_empty(self):
        """Test /api/log/ returns empty lines list when no session running."""
        response = self.client.get(reverse("pipeline_log"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("lines", data)
        self.assertIsInstance(data["lines"], list)

    def test_dataset_preview_no_dataset(self):
        """Test /api/preview/ returns error dict when no dataset in session."""
        response = self.client.get(reverse("dataset_preview"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("columns", data)
        self.assertIn("rows", data)

    def test_dataset_preview_with_real_csv(self):
        """Test /api/preview/ returns rows and columns for a real sample CSV."""
        import os
        from django.conf import settings
        test_dir = settings.TEST_DATA_DIR
        sample_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]
        if not sample_files:
            self.skipTest("No sample CSV files available in test_data/")
        csv_path = os.path.join(test_dir, sample_files[0])
        session = self.client.session
        session["selected_csv_path"] = csv_path
        session.save()
        response = self.client.get(reverse("dataset_preview"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("columns", data)
        self.assertIn("rows", data)
        self.assertIn("total_rows", data)
        self.assertGreater(len(data["columns"]), 0)
        self.assertGreater(len(data["rows"]), 0)

    def test_reset_session(self):
        """Test reset session endpoint clears state and flushes session."""
        response = self.client.post(reverse("reset_session"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "reset")

    def test_run_pipeline_sample_missing(self):
        """Test running pipeline with non-existent sample dataset returns 400."""
        response = self.client.post(reverse("run_pipeline"), {
            "data_source": "sample",
            "sample_file": "non_existent_file_999.csv",
        })
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    def test_run_pipeline_sample_success(self):
        """Test launching pipeline with a valid sample CSV dataset."""
        # Find first sample CSV in TEST_DATA_DIR if available
        test_dir = settings.TEST_DATA_DIR
        sample_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]
        if not sample_files:
            self.skipTest("No sample CSV files available in test_data/")
        
        sample_file = sample_files[0]
        response = self.client.post(reverse("run_pipeline"), {
            "data_source": "sample",
            "sample_file": sample_file,
            "user_request": "Run quick EDA test",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "running")
        self.assertIn("sid", data)

    def test_run_pipeline_upload_missing(self):
        """Test running pipeline with data_source upload but no file uploaded."""
        response = self.client.post(reverse("run_pipeline"), {
            "data_source": "upload",
        })
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    def test_run_pipeline_upload_success(self):
        """Test running pipeline with uploaded CSV file."""
        csv_content = b"col1,col2\n1,a\n2,b\n3,c\n"
        uploaded_file = io.BytesIO(csv_content)
        uploaded_file.name = "unit_test_data.csv"

        response = self.client.post(reverse("run_pipeline"), {
            "data_source": "upload",
            "csv_file": uploaded_file,
            "user_request": "Test uploaded file",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "running")

    def test_submit_answer_without_question(self):
        """Test submitting an answer when no clarifying question is pending."""
        response = self.client.post(reverse("submit_answer"), {
            "answer": "My response to non-existent question"
        })
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    def test_download_report_not_found(self):
        """Test download report when no report exists returns 404."""
        response = self.client.get(reverse("download_report"))
        self.assertEqual(response.status_code, 404)

    def test_serve_artifact_traversal_forbidden(self):
        """Test directory traversal attack prevention in serve_artifact view."""
        response = self.client.get("/artifact/../../manage.py")
        self.assertEqual(response.status_code, 403)

    def test_serve_artifact_not_found(self):
        """Test requesting non-existent artifact returns 404."""
        response = self.client.get("/artifact/non_existent_path_xyz.png")
        self.assertEqual(response.status_code, 404)
