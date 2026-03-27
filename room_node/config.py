"""
Room node configuration loaded from environment / .env file.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class RoomNodeConfig(BaseSettings):
    """All configuration for a room node instance.

    Values are read from environment variables or a .env file in the
    working directory.  See .env.example for documentation on each field.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
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

    # Logging
    log_level: str = "INFO"
