"""Application settings and environment loading."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppSettings:
    """Runtime settings loaded from environment variables."""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    data_dir: str = os.getenv("DATA_DIR", "data/uploads")
    api_url: str = os.getenv("API_URL", "http://localhost:8000")


settings = AppSettings()
