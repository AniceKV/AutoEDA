import os
from typing import Optional

DEFAULT_MODEL = "openrouter/free"   # OpenRouter's own auto-router: picks a live free model per-request
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def get_api_key(override: Optional[str] = None) -> str:
    key = override or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Get a free key at https://openrouter.ai/keys "
            "and set it as an env var, in a .env file, or pass api_key= directly."
        )
    return key


def get_model(override: Optional[str] = None) -> str:
    return override or os.getenv("EDA_MODEL", DEFAULT_MODEL)


def get_base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
