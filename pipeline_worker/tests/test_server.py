"""Tests for pipeline_worker.server — /ingest, /health, /reload_voiceprints.

Uses FastAPI TestClient with a Settings instance pointing at a temp voiceprints
DB and a mocked MemoryClient so no live services are required.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pipeline_worker.models import ConfidenceLevel, EmotionReading
from pipeline_worker.server import AppState, create_app, _process_payload
from pipeline_worker.settings import Settings
from pipeline_worker.voiceprint import VoiceprintMatcher

VOICEPRINT_DIM = 256


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def unit_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(VOICEPRINT_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_mcp_url="http://mock-mcp:8900",
        voiceprint_db=str(tmp_path / "vp.sqlite"),
        pipeline_port=8001,
        ha_webhook_url="",
        hf_token="",
        log_level="WARNING",
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_memory():
    """Return a MemoryClient where all methods are AsyncMocks."""
    m = MagicMock()
    m.record = AsyncMock(return_value={"ok": True})
    m.open_session = AsyncMock(return_value={"result": 42})
    m.log_turn = AsyncMock(return_value={"ok": True})
    m.extract_and_remember = AsyncMock(return_value={"ok": True})
    m.update_voiceprint = AsyncMock(return_value={"ok": True})
    m.aclose = AsyncMock()
    return m


@pytest.fixture
def state(settings, tmp_path, mock_memory):
    st = AppState(settings)
    st.memory = mock_memory
    return st


def make_payload(
    confidence: float = 0.92,
    voiceprint: list | None = None,
    audio_b64: str | None = None,
) -> dict:
    return {
        "room": "kitchen",
        "timestamp": "2026-03-22T10:30:00Z",
        "transcript": "I need to pick up groceries tomorrow",
        "whisper_confidence": confidence,
        "whisper_model": "small",
        "doa": 247,
        "emotion": {"valence": 0.6, "arousal": 0.3},
        "voiceprint": voiceprint or [0.0] * VOICEPRINT_DIM,
        "audio_clip_b64": audio_b64,
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "voiceprints_cached" in data


# ---------------------------------------------------------------------------
# POST /ingest — happy paths
# ---------------------------------------------------------------------------


class TestIngestHappyPath:
    def test_unknown_speaker_no_enrolled_voiceprints(self, client):
        """No enrolled voiceprints → every speaker is unknown."""
        resp = client.post("/ingest", json=make_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["confidence_level"] == "unknown"
        assert data["entity_name"].startswith("unknown_voice_")

    def test_confident_match_returns_entity_name(self, client, app):
        """Pre-load a voiceprint → confident match."""
        state: AppState = app.state.app_state
        v = unit_vec(0)
        state.matcher.upsert("Brian", v)

        payload = make_payload(voiceprint=v.tolist())
        resp = client.post("/ingest", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_name"] == "Brian"
        assert data["confidence_level"] == "confident"

    def test_transcript_in_response(self, client):
        resp = client.post("/ingest", json=make_payload())
        assert resp.json()["transcript"] == "I need to pick up groceries tomorrow"

    def test_payload_without_voiceprint_returns_unknown(self, client):
        payload = make_payload()
        payload["voiceprint"] = None
        resp = client.post("/ingest", json=payload)
        assert resp.status_code == 200
        assert resp.json()["confidence_level"] == "unknown"


# ---------------------------------------------------------------------------
# POST /ingest — memory-mcp writes
# ---------------------------------------------------------------------------


class TestIngestMemoryWrites:
    @pytest.mark.asyncio
    async def test_record_called(self, state):
        from pipeline_worker.models import AudioPayload, EmotionReading
        payload = AudioPayload(
            room="kitchen",
            timestamp="2026-03-22T10:30:00Z",
            transcript="hello world",
            whisper_confidence=0.92,
            voiceprint=[0.0] * VOICEPRINT_DIM,
            emotion=EmotionReading(valence=0.5, arousal=0.5),
        )
        await _process_payload(payload, state)
        state.memory.record.assert_called_once()
        call_kwargs = state.memory.record.call_args
        assert call_kwargs.kwargs["metric"] == "voice_activity"

    @pytest.mark.asyncio
    async def test_open_session_called(self, state):
        from pipeline_worker.models import AudioPayload
        payload = AudioPayload(
            room="office",
            timestamp="2026-03-22T10:30:00Z",
            transcript="hello",
            whisper_confidence=0.92,
            voiceprint=[0.0] * VOICEPRINT_DIM,
        )
        await _process_payload(payload, state)
        state.memory.open_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_turn_called_when_session_opened(self, state):
        from pipeline_worker.models import AudioPayload
        payload = AudioPayload(
            room="kitchen",
            timestamp="2026-03-22T10:30:00Z",
            transcript="hello",
            whisper_confidence=0.92,
            voiceprint=[0.0] * VOICEPRINT_DIM,
        )
        await _process_payload(payload, state)
        # Two log_turn calls per ingest: one "system" context turn + one "user" utterance turn
        assert state.memory.log_turn.call_count == 2

    @pytest.mark.asyncio
    async def test_extract_and_remember_called(self, state):
        from pipeline_worker.models import AudioPayload
        payload = AudioPayload(
            room="kitchen",
            timestamp="2026-03-22T10:30:00Z",
            transcript="I prefer dark roast coffee",
            whisper_confidence=0.92,
            voiceprint=[0.0] * VOICEPRINT_DIM,
        )
        await _process_payload(payload, state)
        state.memory.extract_and_remember.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_voiceprint_called_only_on_confident(self, state):
        from pipeline_worker.models import AudioPayload
        v = unit_vec(0)
        state.matcher.upsert("Brian", v)

        # Confident match
        payload = AudioPayload(
            room="kitchen",
            timestamp="2026-03-22T10:30:00Z",
            transcript="hello",
            whisper_confidence=0.92,
            voiceprint=v.tolist(),
        )
        await _process_payload(payload, state)
        state.memory.update_voiceprint.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_voiceprint_not_called_on_unknown(self, state):
        from pipeline_worker.models import AudioPayload
        payload = AudioPayload(
            room="kitchen",
            timestamp="2026-03-22T10:30:00Z",
            transcript="hello",
            whisper_confidence=0.92,
            voiceprint=[0.0] * VOICEPRINT_DIM,
        )
        await _process_payload(payload, state)
        state.memory.update_voiceprint.assert_not_called()


# ---------------------------------------------------------------------------
# POST /ingest — flags
# ---------------------------------------------------------------------------


class TestIngestFlags:
    def test_unknown_speaker_flag(self, client):
        resp = client.post("/ingest", json=make_payload())
        assert "unknown_speaker" in resp.json()["flags"]

    def test_probable_match_flag(self, client, app):
        state: AppState = app.state.app_state
        stored = unit_vec(0)
        state.matcher.upsert("Brian", stored)

        # Craft embedding with ~75% similarity
        orth = unit_vec(99)
        orth -= orth.dot(stored) * stored
        orth /= np.linalg.norm(orth)
        target_sim = 0.75
        query = (target_sim * stored + np.sqrt(1 - target_sim**2) * orth).astype(np.float32)
        query /= np.linalg.norm(query)

        resp = client.post("/ingest", json=make_payload(voiceprint=query.tolist()))
        data = resp.json()
        assert data["confidence_level"] == "probable"
        assert "probable_match" in data["flags"]


# ---------------------------------------------------------------------------
# POST /ingest — validation errors
# ---------------------------------------------------------------------------


class TestIngestValidation:
    def test_wrong_voiceprint_dim(self, client):
        payload = make_payload(voiceprint=[0.0] * 128)
        resp = client.post("/ingest", json=payload)
        assert resp.status_code == 422

    def test_empty_transcript_rejected(self, client):
        payload = make_payload()
        payload["transcript"] = ""
        resp = client.post("/ingest", json=payload)
        assert resp.status_code == 422

    def test_invalid_base64_rejected(self, client):
        payload = make_payload(audio_b64="not!!valid@@base64")
        resp = client.post("/ingest", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /ingest — capture node payloads
# ---------------------------------------------------------------------------


import base64


def make_capture_payload(audio_b64: str | None = None) -> dict:
    """Build a minimal capture-node payload (no inference fields)."""
    if audio_b64 is None:
        audio_b64 = base64.b64encode(b"RIFF" + b"\x00" * 40).decode()
    return {
        "room": "bedroom",
        "timestamp": "2026-03-22T10:30:00Z",
        "node_profile": "capture",
        "doa": 90,
        "audio_clip_b64": audio_b64,
        "duration_ms": 2500,
    }


class TestIngestCaptureNode:
    def test_capture_payload_accepted(self, client):
        """Pipeline worker accepts capture-node payloads with no transcript."""
        resp = client.post("/ingest", json=make_capture_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

    def test_capture_payload_returns_entity_name(self, client):
        resp = client.post("/ingest", json=make_capture_payload())
        data = resp.json()
        assert "entity_name" in data
        assert data["entity_name"].startswith("unknown_voice_")

    @pytest.mark.asyncio
    async def test_capture_node_skips_extract_and_remember(self, state):
        """No transcript → extract_and_remember is not called."""
        from pipeline_worker.models import AudioPayload
        b64 = base64.b64encode(b"RIFF" + b"\x00" * 40).decode()
        payload = AudioPayload(
            room="bedroom",
            timestamp="2026-03-22T10:30:00Z",
            node_profile="capture",
            audio_clip_b64=b64,
        )
        await _process_payload(payload, state)
        state.memory.extract_and_remember.assert_not_called()


# ---------------------------------------------------------------------------
# POST /reload_voiceprints
# ---------------------------------------------------------------------------


class TestReloadVoiceprints:
    def test_returns_ok(self, client, app):
        state: AppState = app.state.app_state
        # Mock the entities endpoint
        with patch.object(state.memory, "_client") as mock_http:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"result": []}
            mock_http.get = AsyncMock(return_value=mock_response)
            resp = client.post("/reload_voiceprints")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
