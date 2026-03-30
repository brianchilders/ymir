"""
Room node configuration loaded from environment / .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent / ".env"


class RoomNodeConfig(BaseSettings):
    """All configuration for a room node instance.

    Values are read from environment variables or a .env file located in the
    same directory as this file.  See .env.example for documentation on each field.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Room identity
    room_name: str

    # Node profile
    node_profile: Literal["full", "capture"] = "full"

    # Pipeline worker
    blackmagic_url: str = "http://blackmagic.lan:8001"

    # Audio capture
    device_index: int = 0
    sample_rate: int = 16000

    # VAD
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 100
    max_utterance_s: int = 30

    # Confidence thresholds
    whisper_confidence_threshold: float = 0.85

    # Hailo-8
    hailo_enabled: bool = True
    hailo_whisper_hef: str = "./models/whisper_small.hef"
    hailo_emotion_hef: str = "./models/emotion.hef"
    whisper_fallback_model: str = "small"

    # HTTP
    http_max_retries: int = 3
    http_retry_backoff_s: float = 1.0
    offline_queue_maxsize: int = 50

    # Logging
    log_level: str = "INFO"
