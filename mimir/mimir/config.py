"""Mimir configuration loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_MODULE_ENV = Path(__file__).parent.parent / ".env"
_ENV_FILE = str(_MODULE_ENV) if _MODULE_ENV.exists() else ".env"


class MimirConfig(BaseSettings):
    """All configuration for the Mimir intent router."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Upstream services
    ollama_url: str = "http://blackmagic.lan:11434"
    verdandi_url: str = "http://blackmagic.lan:8901"
    muninn_url: str = "http://blackmagic.lan:8900"

    # LLM
    mimir_llm_model: str = "llama3.2:3b"
    mimir_llm_max_tokens: int = 80
    mimir_llm_temperature: float = 0.4

    # Avatar output
    relay_host: str = "192.168.76.195"
    relay_port: int = 8767
    avatar_rooms: str = "living_room,office,kitchen"

    # Silence rules
    silence_cooldown_seconds: int = 90
    greeting_cooldown_minutes: int = 5

    # Service
    mimir_port: int = 8902

    # Logging
    log_level: str = "INFO"

    @property
    def avatar_room_set(self) -> set[str]:
        """Parsed set of room names that have an avatar."""
        return {r.strip() for r in self.avatar_rooms.split(",") if r.strip()}

    @property
    def relay_url(self) -> str:
        """Full URL to the avatar relay server."""
        return f"http://{self.relay_host}:{self.relay_port}/relay"
