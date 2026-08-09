import os
from typing import Optional
from dotenv import load_dotenv, find_dotenv

DEFAULT_MODEL = "google/gemini-3.6-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

_dotenv_loaded = False


def _ensure_dotenv() -> None:
    """Lazily load the nearest .env file once, on first credential access."""
    global _dotenv_loaded
    if not _dotenv_loaded:
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path, override=True)
        _dotenv_loaded = True


def get_api_key(override: Optional[str] = None) -> str:
    """Return the API key: explicit override > env var > error."""
    if override:
        return override
    _ensure_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Get a free key at https://openrouter.ai/keys "
            "and set it as an env var, in a .env file, or pass api_key= directly."
        )
    return key


def get_model(override: Optional[str] = None) -> str:
    """Return the model name: explicit override > env var > default."""
    if override:
        return override
    _ensure_dotenv()
    return os.getenv("EDA_MODEL", DEFAULT_MODEL)


def get_base_url() -> str:
    _ensure_dotenv()
    return os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
