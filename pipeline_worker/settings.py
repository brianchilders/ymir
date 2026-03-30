"""
Pipeline worker configuration loaded from environment / .env file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent / ".env"


class Settings(BaseSettings):
    """All configuration for the pipeline worker.

    Values are read from environment variables or a .env file located in the
    same directory as this file.  See .env.example for documentation on each field.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # memory-mcp
    memory_mcp_url: str = "http://localhost:8900"
    memory_mcp_token: str = ""

    # Service
    pipeline_port: int = 8001

    # Voiceprint matching
    voiceprint_db: str = "./voiceprints.sqlite"
    voiceprint_confident_threshold: float = 0.85
    voiceprint_probable_threshold: float = 0.70

    # Whisper confidence below this → attach audio clip for large-v3 fallback
    whisper_confidence_threshold: float = 0.85

    # HuggingFace token for pyannote model (optional; diarization skipped if absent)
    hf_token: str = ""

    # Home Assistant webhook URL for probable-match notifications (optional)
    ha_webhook_url: str = ""

    # Logging
    log_level: str = "INFO"
