"""
Pydantic v2 data models for the Heimdall audio pipeline.

These models define the canonical wire format between room nodes and the
pipeline worker, as well as internal result types used throughout the service.
"""

from __future__ import annotations

import base64
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

VOICEPRINT_DIM = 256


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConfidenceLevel(str, Enum):
    """Speaker identification confidence tier.

    Maps to cosine-similarity thresholds configured in the pipeline worker.
    CONFIDENT  >= 0.85  — log under entity name, update voiceprint
    PROBABLE   >= 0.70  — log under entity name, send HA notification
    UNKNOWN    <  0.70  — create provisional entity unknown_voice_{hash}
    """

    CONFIDENT = "confident"
    PROBABLE = "probable"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class EmotionReading(BaseModel):
    """Valence/arousal emotion estimate from the on-device emotion model.

    Both axes are normalised to [0, 1] rather than the raw [-1, 1] / [0, 1]
    output of most models, making them comparable across fallback backends.

    valence: 0 = very negative, 1 = very positive
    arousal: 0 = calm/low energy, 1 = excited/high energy
    """

    valence: Annotated[float, Field(ge=0.0, le=1.0)]
    arousal: Annotated[float, Field(ge=0.0, le=1.0)]


# ---------------------------------------------------------------------------
# Wire format: room node → pipeline worker
# ---------------------------------------------------------------------------


class AudioPayload(BaseModel):
    """Structured payload sent by a room node after processing one utterance.

    Two node profiles are supported:

    - ``full`` (Pi 5 + Hailo-8): runs Whisper, emotion, and resemblyzer
      locally.  transcript, whisper_confidence, emotion, and voiceprint are
      all populated.  audio_clip_b64 is attached only when
      whisper_confidence < WHISPER_CONFIDENCE_THRESHOLD.

    - ``capture`` (Pi 4, no AI accelerator): VAD + DOA only.  transcript,
      whisper_confidence, emotion, and voiceprint are all None.  audio_clip_b64
      is always attached — blackmagic runs the full inference stack.
    """

    # --- provenance ---
    room: str = Field(..., description="Room identifier, e.g. 'kitchen'")
    timestamp: datetime = Field(..., description="UTC time of utterance start")

    # --- node profile ---
    node_profile: Literal["full", "capture"] = Field(
        default="full",
        description="Node capability profile: 'full' (on-device inference) or 'capture' (raw audio only)",
    )

    # --- transcription (full node only) ---
    transcript: Optional[str] = Field(
        default=None, min_length=1, description="Whisper transcript text; None for capture nodes"
    )
    whisper_confidence: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default=None, description="Mean segment log-prob converted to [0,1]; None for capture nodes"
    )
    whisper_model: str = Field(default="small", description="Whisper model variant used")

    # --- spatial ---
    doa: Optional[Annotated[int, Field(ge=0, le=359)]] = Field(
        default=None, description="Direction of arrival in degrees (0–359)"
    )

    # --- emotion ---
    emotion: Optional[EmotionReading] = Field(
        default=None, description="Valence/arousal estimate; None if model unavailable"
    )

    # --- identity ---
    voiceprint: Optional[list[float]] = Field(
        default=None,
        description=(
            f"Resemblyzer {VOICEPRINT_DIM}-dim GE2E embedding. "
            "Present for utterances long enough for reliable embedding (>1s). "
            "None for very short or noisy segments."
        ),
    )

    # --- audio clip (capture node: always; full node: low-confidence fallback) ---
    audio_clip_b64: Optional[str] = Field(
        default=None,
        description=(
            "Base64-encoded mono 16kHz PCM WAV clip. "
            "Always present for capture nodes; "
            "attached by full nodes when whisper_confidence < WHISPER_CONFIDENCE_THRESHOLD."
        ),
    )
    duration_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Audio clip duration in milliseconds; present whenever audio_clip_b64 is set",
    )

    # --- validators ---

    @field_validator("voiceprint")
    @classmethod
    def validate_voiceprint_dim(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        """Reject embeddings that are not exactly VOICEPRINT_DIM floats."""
        if v is not None and len(v) != VOICEPRINT_DIM:
            raise ValueError(
                f"voiceprint must be {VOICEPRINT_DIM}-dimensional, got {len(v)}"
            )
        return v

    @field_validator("audio_clip_b64")
    @classmethod
    def validate_base64(cls, v: Optional[str]) -> Optional[str]:
        """Reject audio_clip_b64 values that are not valid base64."""
        if v is not None:
            try:
                base64.b64decode(v, validate=True)
            except Exception as exc:
                raise ValueError("audio_clip_b64 must be valid base64") from exc
        return v

    @field_validator("transcript", mode="before")
    @classmethod
    def strip_transcript(cls, v: object) -> object:
        """Trim leading/trailing whitespace before min_length check fires."""
        return v.strip() if isinstance(v, str) else v


# ---------------------------------------------------------------------------
# Internal: voiceprint matching result
# ---------------------------------------------------------------------------


class VoiceprintMatch(BaseModel):
    """Result of matching an incoming voiceprint against enrolled speakers.

    Returned by VoiceprintMatcher.match().  The entity_name is either a
    resolved enrolled name or a provisional 'unknown_voice_{hash}' string.
    """

    entity_name: str = Field(..., description="Resolved or provisional entity name")
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ..., description="Cosine similarity score (0.0 if no candidates)"
    )
    confidence_level: ConfidenceLevel


# ---------------------------------------------------------------------------
# Pipeline worker HTTP response
# ---------------------------------------------------------------------------


class PipelineResponse(BaseModel):
    """Response returned by POST /ingest.

    ok=True even for unknown/probable matches — those are not errors.
    flags carries informational strings the caller may act on
    (e.g. 'probable_match', 'fallback_transcription_used').
    """

    ok: bool = True
    entity_name: str = Field(..., description="Speaker identity used for logging")
    confidence_level: ConfidenceLevel
    transcript: Optional[str] = Field(
        default=None,
        description="Final transcript (may differ from payload if fallback used; None if all transcription failed)",
    )
    session_id: Optional[str] = Field(
        default=None, description="memory-mcp session ID for this conversation turn"
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Informational flags: 'probable_match', 'fallback_transcription_used', etc.",
    )
