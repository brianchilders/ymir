"""Tests for room_node.sender — payload building and HTTP dispatch.

No live pipeline worker required — httpx calls are intercepted with respx.
"""

from __future__ import annotations

import base64
import io
import wave
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import respx
from httpx import Response

from room_node.sender import PayloadSender, _encode_audio, decode_audio

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_audio(duration_s: float = 1.0) -> np.ndarray:
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t) * 0.5


def unit_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(256).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def sender() -> PayloadSender:
    return PayloadSender(
        blackmagic_url="http://blackmagic.test:8001",
        room_name="kitchen",
        node_profile="full",
        whisper_confidence_threshold=0.85,
        max_retries=2,
        retry_backoff_s=0.01,
    )


@pytest.fixture
def capture_sender() -> PayloadSender:
    return PayloadSender(
        blackmagic_url="http://blackmagic.test:8001",
        room_name="bedroom",
        node_profile="capture",
        max_retries=2,
        retry_backoff_s=0.01,
    )


OK_RESPONSE = Response(200, json={"ok": True, "entity_name": "Brian", "confidence_level": "confident", "transcript": "hello", "flags": []})


# ---------------------------------------------------------------------------
# _encode_audio / decode_audio
# ---------------------------------------------------------------------------


class TestAudioEncoding:
    def test_roundtrip(self):
        audio = make_audio(1.0)
        b64 = _encode_audio(audio, SAMPLE_RATE)
        recovered = decode_audio(b64, SAMPLE_RATE)
        # Allow small quantisation error from int16 conversion
        assert np.allclose(audio, recovered, atol=1e-4)

    def test_output_is_string(self):
        b64 = _encode_audio(make_audio(0.5), SAMPLE_RATE)
        assert isinstance(b64, str)

    def test_output_is_valid_base64(self):
        b64 = _encode_audio(make_audio(0.5), SAMPLE_RATE)
        decoded = base64.b64decode(b64)  # should not raise
        assert len(decoded) > 0

    def test_decoded_is_valid_wav(self):
        b64 = _encode_audio(make_audio(0.5), SAMPLE_RATE)
        raw = base64.b64decode(b64)
        buf = io.BytesIO(raw)
        with wave.open(buf, "rb") as wf:
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 16-bit

    def test_wrong_sample_rate_raises(self):
        b64 = _encode_audio(make_audio(0.5), SAMPLE_RATE)
        with pytest.raises(ValueError, match="Hz"):
            decode_audio(b64, expected_sample_rate=8000)

    def test_silence_encodes_without_error(self):
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        b64 = _encode_audio(silence, SAMPLE_RATE)
        assert len(b64) > 0


# ---------------------------------------------------------------------------
# PayloadSender._build_payload
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_high_confidence_no_audio_clip(self, sender):
        payload = sender._build_payload(
            audio=make_audio(),
            doa=180,
            transcript="hello",
            whisper_confidence=0.92,
            whisper_model="small",
            emotion_valence=0.5,
            emotion_arousal=0.5,
            voiceprint=unit_vec(),
        )
        assert payload["audio_clip_b64"] is None

    def test_low_confidence_includes_audio_clip(self, sender):
        payload = sender._build_payload(
            audio=make_audio(),
            doa=None,
            transcript="hello",
            whisper_confidence=0.60,
            whisper_model="small",
            emotion_valence=0.5,
            emotion_arousal=0.5,
            voiceprint=None,
        )
        assert payload["audio_clip_b64"] is not None
        # Verify it decodes as a valid WAV
        decode_audio(payload["audio_clip_b64"], SAMPLE_RATE)

    def test_room_name_in_payload(self, sender):
        payload = sender._build_payload(
            audio=make_audio(),
            doa=90,
            transcript="hello",
            whisper_confidence=0.92,
            whisper_model="small",
            emotion_valence=0.5,
            emotion_arousal=0.5,
            voiceprint=None,
        )
        assert payload["room"] == "kitchen"

    def test_voiceprint_as_list(self, sender):
        vp = unit_vec(0)
        payload = sender._build_payload(
            audio=make_audio(),
            doa=None,
            transcript="hello",
            whisper_confidence=0.92,
            whisper_model="small",
            emotion_valence=0.5,
            emotion_arousal=0.5,
            voiceprint=vp,
        )
        assert isinstance(payload["voiceprint"], list)
        assert len(payload["voiceprint"]) == 256

    def test_none_voiceprint_stays_none(self, sender):
        payload = sender._build_payload(
            audio=make_audio(),
            doa=None,
            transcript="hello",
            whisper_confidence=0.92,
            whisper_model="small",
            emotion_valence=0.5,
            emotion_arousal=0.5,
            voiceprint=None,
        )
        assert payload["voiceprint"] is None

    def test_emotion_fields(self, sender):
        payload = sender._build_payload(
            audio=make_audio(),
            doa=None,
            transcript="hello",
            whisper_confidence=0.92,
            whisper_model="small",
            emotion_valence=0.7,
            emotion_arousal=0.4,
            voiceprint=None,
        )
        assert payload["emotion"]["valence"] == 0.7
        assert payload["emotion"]["arousal"] == 0.4

    def test_timestamp_is_string(self, sender):
        payload = sender._build_payload(
            audio=make_audio(),
            doa=None,
            transcript="hello",
            whisper_confidence=0.92,
            whisper_model="small",
            emotion_valence=0.5,
            emotion_arousal=0.5,
            voiceprint=None,
        )
        assert isinstance(payload["timestamp"], str)

    def test_node_profile_in_full_payload(self, sender):
        payload = sender._build_payload(
            audio=make_audio(),
            doa=None,
            transcript="hello",
            whisper_confidence=0.92,
        )
        assert payload["node_profile"] == "full"


class TestBuildPayloadCaptureNode:
    def test_always_includes_audio_clip(self, capture_sender):
        payload = capture_sender._build_payload(audio=make_audio(), doa=90)
        assert payload["audio_clip_b64"] is not None
        decode_audio(payload["audio_clip_b64"], SAMPLE_RATE)

    def test_no_transcript_field(self, capture_sender):
        payload = capture_sender._build_payload(audio=make_audio(), doa=None)
        assert "transcript" not in payload

    def test_no_emotion_field(self, capture_sender):
        payload = capture_sender._build_payload(audio=make_audio(), doa=None)
        assert "emotion" not in payload

    def test_no_voiceprint_field(self, capture_sender):
        payload = capture_sender._build_payload(audio=make_audio(), doa=None)
        assert "voiceprint" not in payload

    def test_node_profile_is_capture(self, capture_sender):
        payload = capture_sender._build_payload(audio=make_audio(), doa=None)
        assert payload["node_profile"] == "capture"

    def test_duration_ms_computed(self, capture_sender):
        audio = make_audio(2.0)  # 2 seconds
        payload = capture_sender._build_payload(audio=audio, doa=None)
        assert payload["duration_ms"] == pytest.approx(2000, abs=50)

    def test_room_name_correct(self, capture_sender):
        payload = capture_sender._build_payload(audio=make_audio(), doa=None)
        assert payload["room"] == "bedroom"


# ---------------------------------------------------------------------------
# PayloadSender.send — HTTP dispatch
# ---------------------------------------------------------------------------


class TestSend:
    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_delivery(self, sender):
        respx.post("http://blackmagic.test:8001/ingest").mock(return_value=OK_RESPONSE)
        result = await sender.send(
            audio=make_audio(),
            doa=247,
            transcript="hello",
            whisper_confidence=0.92,
            whisper_model="small",
            emotion_valence=0.5,
            emotion_arousal=0.5,
            voiceprint=unit_vec(),
        )
        assert result is not None
        assert result["ok"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_retries_on_connection_error(self, sender):
        import httpx
        route = respx.post("http://blackmagic.test:8001/ingest")
        route.side_effect = [
            httpx.ConnectError("refused"),
            OK_RESPONSE,
        ]
        result = await sender.send(
            audio=make_audio(),
            doa=None,
            transcript="hello",
            whisper_confidence=0.92,
            whisper_model="small",
            emotion_valence=0.5,
            emotion_arousal=0.5,
            voiceprint=None,
        )
        assert result is not None
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_none_after_max_retries(self, sender):
        import httpx
        respx.post("http://blackmagic.test:8001/ingest").mock(
            side_effect=httpx.ConnectError("refused")
        )
        result = await sender.send(
            audio=make_audio(),
            doa=None,
            transcript="hello",
            whisper_confidence=0.92,
            whisper_model="small",
            emotion_valence=0.5,
            emotion_arousal=0.5,
            voiceprint=None,
        )
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_server_error_returns_none_after_retries(self, sender):
        respx.post("http://blackmagic.test:8001/ingest").mock(
            return_value=Response(500, text="Internal Error")
        )
        result = await sender.send(
            audio=make_audio(),
            doa=None,
            transcript="hello",
            whisper_confidence=0.92,
            whisper_model="small",
            emotion_valence=0.5,
            emotion_arousal=0.5,
            voiceprint=None,
        )
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_capture_sender_delivers(self, capture_sender):
        """Capture node send() requires only audio and doa."""
        respx.post("http://blackmagic.test:8001/ingest").mock(return_value=OK_RESPONSE)
        result = await capture_sender.send(audio=make_audio(), doa=90)
        assert result is not None
        assert result["ok"] is True
