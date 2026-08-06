import os
from pathlib import Path

# Base directory of the Django project (django_app/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Root of the entire AutoEDA repo (one level up from django_app/)
AUTOEDA_ROOT = BASE_DIR.parent

SECRET_KEY = "django-autoeda-pro-secret-key-change-in-production-xk92pl"

DEBUG = True

ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "eda_app",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

CSRF_TRUSTED_ORIGINS = [
    "https://*.onrender.com",
    "https://*.railway.app",
    "https://*.up.railway.app",
    "http://*",
    "https://*",
]

X_FRAME_OPTIONS = "SAMEORIGIN"

ROOT_URLCONF = "autoeda.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "autoeda.wsgi.application"

# In-memory SQLite for test runner and Django core compatibility
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

# Session backend: file-based (no DB required)
SESSION_ENGINE = "django.contrib.sessions.backends.file"
SESSION_FILE_PATH = os.path.join(BASE_DIR, ".sessions")
os.makedirs(SESSION_FILE_PATH, exist_ok=True)
SESSION_COOKIE_AGE = 86400  # 24 hours

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = False

# Static files
STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# Media / artifact files served from AUTOEDA_ROOT
MEDIA_URL = "/media/"
MEDIA_ROOT = str(AUTOEDA_ROOT)

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Path to test_data directory for sample datasets
TEST_DATA_DIR = os.path.join(str(AUTOEDA_ROOT), "test_data")
TEMP_UPLOADS_DIR = os.path.join(str(AUTOEDA_ROOT), "temp_uploads")
SANDBOX_DIR = os.path.join(str(AUTOEDA_ROOT), "sandbox_run")
EDA_DIR = os.path.join(str(AUTOEDA_ROOT), "EDA")
