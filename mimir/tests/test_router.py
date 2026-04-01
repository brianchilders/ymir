"""
Unit tests for mimir.router.

All Ollama, Verdandi, and Muninn calls are mocked.
pyttsx3 is not required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nornir.models import ContextEvent, ScoredMemory
from mimir.config import MimirConfig
from mimir.cooldown import CooldownState
from mimir.router import _build_prompt, _is_silent, route

FIXTURES = Path(__file__).parent / "fixtures" / "sample_routes.json"

_CONFIG = MimirConfig(
    ollama_url="http://localhost:11434",
    verdandi_url="http://localhost:8901",
    muninn_url="http://localhost:8900",
    mimir_llm_model="llama3.2:3b",
    mimir_llm_max_tokens=80,
    mimir_llm_temperature=0.4,
    relay_host="192.168.1.100",
    relay_port=8767,
    avatar_rooms="living_room,office,kitchen",
    silence_cooldown_seconds=90,
    greeting_cooldown_minutes=5,
)

_FRESH_COOLDOWN = lambda: CooldownState(cooldown_s=90, greeting_cooldown_s=300)


def _event(transcript: str = "hello", location: str = "kitchen", who: str = "Brian") -> ContextEvent:
    return ContextEvent(
        who=who,
        transcript=transcript,
        emotion="neutral",
        location=location,
        local_time="2026-04-01T08:00:00",
    )


def _memory(i: int = 0, score: float = 0.8) -> ScoredMemory:
    return ScoredMemory(
        id=f"mem-{i}",
        content=f"Memory content {i}",
        score=score,
        similarity=0.7,
        recency=0.8,
        urgency=0.0,
        meta={"tier": "semantic"},
    )


def _llm_client(response_text: str) -> AsyncMock:
    """Build a mock AsyncClient that returns a fixed LLM response."""
    llm_resp = MagicMock()
    llm_resp.raise_for_status = MagicMock()
    llm_resp.json.return_value = {"response": response_text}

    followup_resp = MagicMock()
    followup_resp.raise_for_status = MagicMock()
    followup_resp.json.return_value = {"id": "fu-1", "who": "Brian"}

    async def fake_post(url, **kwargs):
        if "generate" in url:
            return llm_resp
        if "followup" in url:
            return followup_resp
        raise ValueError(f"Unexpected POST to {url}")

    client = AsyncMock()
    client.post = fake_post
    return client


# ---------------------------------------------------------------------------
# _is_silent
# ---------------------------------------------------------------------------


def test_is_silent_empty_string():
    assert _is_silent("") is True


def test_is_silent_whitespace():
    assert _is_silent("   ") is True


def test_is_silent_sentinel():
    assert _is_silent("SILENT") is True


def test_is_silent_sentinel_lowercase():
    assert _is_silent("silent") is True


def test_is_silent_normal_text():
    assert _is_silent("Hey Brian, don't forget your appointment.") is False


def test_is_silent_normal_text_with_newline():
    assert _is_silent("Hey Brian, you have a meeting.") is False


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_contains_event_fields():
    event = _event("I need to pick up the kids")
    memories = [_memory(0), _memory(1)]
    system, user = _build_prompt(event, memories, "reminder")

    assert "Brian" in user
    assert "kitchen" in user
    assert "pick up the kids" in user
    assert "reminder" in system
    assert "SILENT" in system


def test_build_prompt_formats_memories():
    memories = [_memory(0, score=0.9), _memory(1, score=0.7)]
    _, user = _build_prompt(_event(), memories, "general")
    assert "Memory content 0" in user
    assert "Memory content 1" in user
    assert "0.90" in user or "0.9" in user


def test_build_prompt_no_memories_shows_none():
    _, user = _build_prompt(_event(), [], "general")
    assert "(none)" in user


def test_build_prompt_system_has_rules():
    system, _ = _build_prompt(_event(), [], "general")
    assert "1-2 sentences" in system
    assert "address the person" in system.lower() or "address" in system.lower()


# ---------------------------------------------------------------------------
# route — silent paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_silent_when_cooldown_active():
    cooldown = _fresh = CooldownState(cooldown_s=90, greeting_cooldown_s=300)
    cooldown.record_speech("general")
    client = AsyncMock()

    result = await route(
        event=_event(),
        memories=[_memory()],
        config=_CONFIG,
        cooldown=cooldown,
        client=client,
        use_avatar=False,
    )
    assert result is None
    # No LLM call should have been made
    client.post.assert_not_called()


@pytest.mark.asyncio
async def test_route_silent_when_no_memories():
    cooldown = _FRESH_COOLDOWN()
    client = AsyncMock()

    result = await route(
        event=_event(),
        memories=[],
        config=_CONFIG,
        cooldown=cooldown,
        client=client,
        use_avatar=False,
    )
    assert result is None
    client.post.assert_not_called()


@pytest.mark.asyncio
async def test_route_silent_when_llm_returns_silent():
    cooldown = _FRESH_COOLDOWN()
    client = _llm_client("")  # empty = SILENT

    with patch("mimir.router.deliver_tts", new_callable=AsyncMock):
        result = await route(
            event=_event(),
            memories=[_memory()],
            config=_CONFIG,
            cooldown=cooldown,
            client=client,
            use_avatar=False,
        )
    assert result is None


# ---------------------------------------------------------------------------
# route — spoken paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_returns_routing_result_when_spoken():
    cooldown = _FRESH_COOLDOWN()
    client = _llm_client("Hey Brian, don't forget to pick up the kids at 3pm.")

    with patch("mimir.router.deliver_tts", new_callable=AsyncMock) as mock_tts:
        result = await route(
            event=_event("don't forget my appointment tomorrow"),
            memories=[_memory()],
            config=_CONFIG,
            cooldown=cooldown,
            client=client,
            use_avatar=False,
        )

    assert result is not None
    assert result.spoken_text == "Hey Brian, don't forget to pick up the kids at 3pm."
    assert result.domain == "reminder"
    assert result.output_path == "tts_fallback"
    assert result.latency_ms >= 0
    assert "mem-0" in result.memories_used
    mock_tts.assert_called_once()


@pytest.mark.asyncio
async def test_route_records_speech_in_cooldown():
    cooldown = _FRESH_COOLDOWN()
    client = _llm_client("Brian, your dentist appointment is today.")

    with patch("mimir.router.deliver_tts", new_callable=AsyncMock):
        await route(
            event=_event("appointment"),
            memories=[_memory()],
            config=_CONFIG,
            cooldown=cooldown,
            client=client,
            use_avatar=False,
        )

    assert cooldown.is_silenced("general") is True


@pytest.mark.asyncio
async def test_route_uses_avatar_for_avatar_room():
    cooldown = _FRESH_COOLDOWN()

    avatar_resp = MagicMock()
    avatar_resp.raise_for_status = MagicMock()

    followup_resp = MagicMock()
    followup_resp.raise_for_status = MagicMock()
    followup_resp.json.return_value = {"id": "fu-1", "who": "Brian"}

    llm_resp = MagicMock()
    llm_resp.raise_for_status = MagicMock()
    llm_resp.json.return_value = {"response": "Hey Brian, something important."}

    async def fake_post(url, **kwargs):
        if "generate" in url:
            return llm_resp
        if "relay" in url:
            return avatar_resp
        if "followup" in url:
            return followup_resp
        raise ValueError(f"Unexpected POST {url}")

    client = AsyncMock()
    client.post = fake_post

    with patch("mimir.router.deliver_tts", new_callable=AsyncMock):
        result = await route(
            event=_event(location="living_room"),
            memories=[_memory()],
            config=_CONFIG,
            cooldown=cooldown,
            client=client,
            use_avatar=True,
        )

    assert result is not None
    assert result.output_path == "avatar"


@pytest.mark.asyncio
async def test_route_falls_back_to_tts_when_avatar_fails():
    cooldown = _FRESH_COOLDOWN()

    llm_resp = MagicMock()
    llm_resp.raise_for_status = MagicMock()
    llm_resp.json.return_value = {"response": "Something important."}

    followup_resp = MagicMock()
    followup_resp.raise_for_status = MagicMock()
    followup_resp.json.return_value = {"id": "fu-1", "who": "Brian"}

    async def fake_post(url, **kwargs):
        if "generate" in url:
            return llm_resp
        if "relay" in url:
            raise httpx_error()
        if "followup" in url:
            return followup_resp
        raise ValueError(url)

    import httpx

    def httpx_error():
        return httpx.ConnectError("refused")

    client = AsyncMock()
    client.post = fake_post

    with patch("mimir.router.deliver_tts", new_callable=AsyncMock) as mock_tts:
        result = await route(
            event=_event(location="living_room"),
            memories=[_memory()],
            config=_CONFIG,
            cooldown=cooldown,
            client=client,
            use_avatar=True,
        )

    assert result is not None
    assert result.output_path == "tts_fallback"
    mock_tts.assert_called_once()


@pytest.mark.asyncio
async def test_route_does_not_use_avatar_for_non_avatar_room():
    cooldown = _FRESH_COOLDOWN()
    client = _llm_client("Something to say.")

    with patch("mimir.router.deliver_tts", new_callable=AsyncMock):
        result = await route(
            event=_event(location="garage"),  # not in avatar_rooms
            memories=[_memory()],
            config=_CONFIG,
            cooldown=cooldown,
            client=client,
            use_avatar=True,
        )

    assert result is not None
    assert result.output_path == "tts_fallback"
