import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
django_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "django_app"))
if django_app_path not in sys.path:
    sys.path.insert(0, django_app_path)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autoeda.settings")
import django
django.setup()

from eda_app.views import _load_metrics, index
from django.test import RequestFactory


def test_views_hypothesis_parsing_per_feature():
    metrics = {
        "statistical_hypothesis_tests": {
            "gender": {
                "test_name": "Two-Sample Welch T-Test",
                "statistic": 9.9977,
                "p_value": 1.7118e-22,
                "is_statistically_significant": True,
                "interpretation": "Significant gap"
            },
            "significant_predictors": ["gender"]
        }
    }
    factory = RequestFactory()
    request = factory.get("/")
    request.session = {}

    from unittest.mock import patch
    with patch("eda_app.views._load_metrics", return_value=metrics):
        response = index(request)
        assert response.status_code == 200


def test_views_hypothesis_parsing_ranked_details():
    metrics = {
        "statistical_hypothesis_tests": {
            "target_col": "writing score",
            "ranked_significant_details": [
                {
                    "feature": "reading score",
                    "test": "Pearson Correlation",
                    "effect_size": 0.9546,
                    "p_value": 0.0,
                    "is_statistically_significant": True,
                    "interpretation": "High correlation"
                }
            ],
            "significant_predictors": ["reading score"]
        }
    }
    factory = RequestFactory()
    request = factory.get("/")
    request.session = {}

    from unittest.mock import patch
    with patch("eda_app.views._load_metrics", return_value=metrics):
        response = index(request)
        assert response.status_code == 200
