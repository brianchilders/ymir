"""Tests for pipeline_worker.models."""

import base64

import pytest
from pydantic import ValidationError

from pipeline_worker.models import (
    VOICEPRINT_DIM,
    AudioPayload,
    ConfidenceLevel,
    EmotionReading,
    PipelineResponse,
    VoiceprintMatch,
)


# ---------------------------------------------------------------------------
# EmotionReading
# ---------------------------------------------------------------------------


class TestEmotionReading:
    def test_valid(self):
        e = EmotionReading(valence=0.6, arousal=0.3)
        assert e.valence == 0.6
        assert e.arousal == 0.3

    def test_boundary_values(self):
        EmotionReading(valence=0.0, arousal=0.0)
        EmotionReading(valence=1.0, arousal=1.0)

    def test_valence_out_of_range(self):
        with pytest.raises(ValidationError):
            EmotionReading(valence=1.1, arousal=0.5)

    def test_arousal_out_of_range(self):
        with pytest.raises(ValidationError):
            EmotionReading(valence=0.5, arousal=-0.1)


# ---------------------------------------------------------------------------
# AudioPayload
# ---------------------------------------------------------------------------


class TestAudioPayload:
    BASE = dict(
        room="kitchen",
        timestamp="2026-03-22T10:30:00Z",
        transcript="hello world",
        whisper_confidence=0.92,
    )

    def test_minimal_valid(self):
        p = AudioPayload(**self.BASE)
        assert p.room == "kitchen"
        assert p.whisper_model == "small"  # default
        assert p.doa is None
        assert p.emotion is None
        assert p.voiceprint is None
        assert p.audio_clip_b64 is None

    def test_full_payload(self):
        p = AudioPayload(
            **self.BASE,
            whisper_model="medium",
            doa=180,
            emotion=EmotionReading(valence=0.5, arousal=0.5),
            voiceprint=[0.01] * VOICEPRINT_DIM,
        )
        assert p.doa == 180
        assert len(p.voiceprint) == VOICEPRINT_DIM

    def test_transcript_stripped(self):
        p = AudioPayload(**{**self.BASE, "transcript": "  hello  "})
        assert p.transcript == "hello"

    def test_empty_transcript_rejected(self):
        with pytest.raises(ValidationError):
            AudioPayload(**{**self.BASE, "transcript": ""})

    def test_whitespace_only_transcript_rejected(self):
        with pytest.raises(ValidationError):
            AudioPayload(**{**self.BASE, "transcript": "   "})

    def test_whisper_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            AudioPayload(**{**self.BASE, "whisper_confidence": 1.5})

    def test_doa_boundary_values(self):
        AudioPayload(**{**self.BASE, "doa": 0})
        AudioPayload(**{**self.BASE, "doa": 359})

    def test_doa_out_of_range(self):
        with pytest.raises(ValidationError):
            AudioPayload(**{**self.BASE, "doa": 360})
        with pytest.raises(ValidationError):
            AudioPayload(**{**self.BASE, "doa": -1})

    def test_voiceprint_wrong_dim(self):
        with pytest.raises(ValidationError, match="256-dimensional"):
            AudioPayload(**{**self.BASE, "voiceprint": [0.1] * 128})

    def test_voiceprint_correct_dim(self):
        p = AudioPayload(**{**self.BASE, "voiceprint": [0.0] * VOICEPRINT_DIM})
        assert len(p.voiceprint) == VOICEPRINT_DIM

    def test_audio_clip_valid_base64(self):
        b64 = base64.b64encode(b"RIFF" + b"\x00" * 40).decode()
        p = AudioPayload(**{**self.BASE, "audio_clip_b64": b64})
        assert p.audio_clip_b64 == b64

    def test_audio_clip_invalid_base64(self):
        with pytest.raises(ValidationError, match="valid base64"):
            AudioPayload(**{**self.BASE, "audio_clip_b64": "not!!base64@@"})

    def test_node_profile_default_is_full(self):
        p = AudioPayload(**self.BASE)
        assert p.node_profile == "full"

    def test_node_profile_explicit_capture(self):
        b64 = base64.b64encode(b"RIFF" + b"\x00" * 40).decode()
        p = AudioPayload(
            room="bedroom",
            timestamp="2026-03-22T10:30:00Z",
            node_profile="capture",
            audio_clip_b64=b64,
            duration_ms=2500,
        )
        assert p.node_profile == "capture"
        assert p.transcript is None
        assert p.whisper_confidence is None
        assert p.emotion is None
        assert p.voiceprint is None
        assert p.duration_ms == 2500

    def test_capture_node_transcript_none_allowed(self):
        """Capture nodes do not produce transcripts — None must be valid."""
        b64 = base64.b64encode(b"RIFF" + b"\x00" * 40).decode()
        p = AudioPayload(
            room="bedroom",
            timestamp="2026-03-22T10:30:00Z",
            node_profile="capture",
            audio_clip_b64=b64,
        )
        assert p.transcript is None

    def test_capture_node_whisper_confidence_none_allowed(self):
        b64 = base64.b64encode(b"RIFF" + b"\x00" * 40).decode()
        p = AudioPayload(
            room="bedroom",
            timestamp="2026-03-22T10:30:00Z",
            node_profile="capture",
            audio_clip_b64=b64,
        )
        assert p.whisper_confidence is None

    def test_node_profile_invalid_value_rejected(self):
        with pytest.raises(ValidationError):
            AudioPayload(**{**self.BASE, "node_profile": "unknown"})


# ---------------------------------------------------------------------------
# VoiceprintMatch
# ---------------------------------------------------------------------------


class TestVoiceprintMatch:
    def test_confident(self):
        m = VoiceprintMatch(
            entity_name="Brian",
            confidence=0.92,
            confidence_level=ConfidenceLevel.CONFIDENT,
        )
        assert m.entity_name == "Brian"
        assert m.confidence_level == ConfidenceLevel.CONFIDENT

    def test_unknown_provisional_name(self):
        m = VoiceprintMatch(
            entity_name="unknown_voice_a3f2c8d1",
            confidence=0.0,
            confidence_level=ConfidenceLevel.UNKNOWN,
        )
        assert m.entity_name.startswith("unknown_voice_")

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            VoiceprintMatch(
                entity_name="Brian",
                confidence=1.1,
                confidence_level=ConfidenceLevel.CONFIDENT,
            )


# ---------------------------------------------------------------------------
# PipelineResponse
# ---------------------------------------------------------------------------


class TestPipelineResponse:
    def test_defaults(self):
        r = PipelineResponse(
            entity_name="Brian",
            confidence_level=ConfidenceLevel.CONFIDENT,
            transcript="hello",
        )
        assert r.ok is True
        assert r.flags == []
        assert r.session_id is None

    def test_with_flags(self):
        r = PipelineResponse(
            entity_name="unknown_voice_abc",
            confidence_level=ConfidenceLevel.UNKNOWN,
            transcript="hello",
            flags=["fallback_transcription_used"],
        )
        assert "fallback_transcription_used" in r.flags
