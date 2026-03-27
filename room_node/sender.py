"""
Payload packaging and HTTP dispatch to the pipeline worker.

Assembles an AudioPayload dict from inference results, attaches a base64
audio clip when Whisper confidence is below threshold, then POSTs to
blackmagic.lan:8001/ingest with exponential-backoff retry.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import wave
from datetime import datetime, timezone
from typing import Literal, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class PayloadSender:
    """Build and dispatch AudioPayload to the pipeline worker.

    Supports both node profiles:

    - ``full``: runs local inference; transcript/emotion/voiceprint are
      populated.  audio_clip_b64 is attached only when
      whisper_confidence < whisper_confidence_threshold.

    - ``capture``: no local inference; audio_clip_b64 is always attached.
      transcript, emotion, and voiceprint are omitted from the payload.

    Usage::

        # Full node
        sender = PayloadSender(
            blackmagic_url="http://blackmagic.lan:8001",
            room_name="kitchen",
            node_profile="full",
            whisper_confidence_threshold=0.85,
        )
        await sender.send(
            audio=utterance_array,
            doa=247,
            transcript="hello world",
            whisper_confidence=0.92,
            whisper_model="small",
            emotion_valence=0.6,
            emotion_arousal=0.3,
            voiceprint=embedding_array,
        )

        # Capture node
        sender = PayloadSender(
            blackmagic_url="http://blackmagic.lan:8001",
            room_name="bedroom",
            node_profile="capture",
        )
        await sender.send(audio=utterance_array, doa=90)
    """

    def __init__(
        self,
        blackmagic_url: str,
        room_name: str,
        node_profile: Literal["full", "capture"] = "full",
        whisper_confidence_threshold: float = 0.85,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
        sample_rate: int = 16000,
    ) -> None:
        self.blackmagic_url = blackmagic_url.rstrip("/")
        self.room_name = room_name
        self.node_profile = node_profile
        self.whisper_confidence_threshold = whisper_confidence_threshold
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.sample_rate = sample_rate

    async def send(
        self,
        audio: np.ndarray,
        doa: Optional[int] = None,
        transcript: Optional[str] = None,
        whisper_confidence: Optional[float] = None,
        whisper_model: str = "small",
        emotion_valence: Optional[float] = None,
        emotion_arousal: Optional[float] = None,
        voiceprint: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """Package and POST one utterance to the pipeline worker.

        For full nodes: attaches audio_clip_b64 when whisper_confidence is
        below threshold.  For capture nodes: always attaches audio_clip_b64
        and omits inference fields.

        Args:
            audio: float32 mono utterance array at sample_rate Hz.
            doa: Direction of arrival in degrees, or None.
            transcript: Whisper transcript string (full node only).
            whisper_confidence: Confidence score [0, 1] (full node only).
            whisper_model: Model variant string, e.g. 'small'.
            emotion_valence: Valence [0, 1] (full node only).
            emotion_arousal: Arousal [0, 1] (full node only).
            voiceprint: 256-dim unit-norm embedding, or None.

        Returns:
            Parsed JSON response from the pipeline worker, or None on failure.
        """
        payload = self._build_payload(
            audio=audio,
            doa=doa,
            transcript=transcript,
            whisper_confidence=whisper_confidence,
            whisper_model=whisper_model,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
            voiceprint=voiceprint,
        )
        return await self._post_with_retry(payload)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        audio: np.ndarray,
        doa: Optional[int],
        transcript: Optional[str] = None,
        whisper_confidence: Optional[float] = None,
        whisper_model: str = "small",
        emotion_valence: Optional[float] = None,
        emotion_arousal: Optional[float] = None,
        voiceprint: Optional[np.ndarray] = None,
    ) -> dict:
        """Assemble the JSON payload dict.

        Capture node: always includes audio_clip_b64; omits inference fields.
        Full node: includes inference fields; includes audio_clip_b64 only
        when whisper_confidence < whisper_confidence_threshold.
        """
        duration_ms = int(len(audio) / self.sample_rate * 1000)

        payload: dict = {
            "room": self.room_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_profile": self.node_profile,
            "doa": doa,
            "audio_clip_b64": None,
        }

        if self.node_profile == "capture":
            # Capture node: always ship raw audio; no inference fields
            payload["audio_clip_b64"] = _encode_audio(audio, self.sample_rate)
            payload["duration_ms"] = duration_ms
            logger.debug("Capture node — attaching audio clip (%d ms)", duration_ms)
        else:
            # Full node: include inference results
            payload["transcript"] = transcript
            payload["whisper_confidence"] = whisper_confidence
            payload["whisper_model"] = whisper_model
            payload["voiceprint"] = voiceprint.tolist() if voiceprint is not None else None
            if emotion_valence is not None and emotion_arousal is not None:
                payload["emotion"] = {"valence": emotion_valence, "arousal": emotion_arousal}

            if whisper_confidence is not None and whisper_confidence < self.whisper_confidence_threshold:
                payload["audio_clip_b64"] = _encode_audio(audio, self.sample_rate)
                payload["duration_ms"] = duration_ms
                logger.debug(
                    "Low Whisper confidence (%.2f) — attaching audio clip for fallback",
                    whisper_confidence,
                )

        return payload

    async def _post_with_retry(self, payload: dict) -> Optional[dict]:
        """POST payload to /ingest with exponential-backoff retry.

        Args:
            payload: JSON-serialisable dict to POST.

        Returns:
            Parsed JSON response, or None if all retries exhausted.
        """
        url = f"{self.blackmagic_url}/ingest"
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as http:
                    response = await http.post(url, json=payload)
                    response.raise_for_status()
                    logger.debug("Payload delivered on attempt %d", attempt)
                    return response.json()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Pipeline worker returned %d (attempt %d/%d): %s",
                    exc.response.status_code,
                    attempt,
                    self.max_retries,
                    exc.response.text[:200],
                )
            except httpx.RequestError as exc:
                logger.warning(
                    "Connection error (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

            if attempt < self.max_retries:
                backoff = self.retry_backoff_s * (2 ** (attempt - 1))
                logger.debug("Retrying in %.1fs", backoff)
                await asyncio.sleep(backoff)

        logger.error("All %d delivery attempts failed for room=%s", self.max_retries, self.room_name)
        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _encode_audio(audio: np.ndarray, sample_rate: int) -> str:
    """Encode a float32 mono audio array as a base64 WAV string.

    Args:
        audio: float32 mono array normalised to [-1, 1].
        sample_rate: Sample rate in Hz.

    Returns:
        Base64-encoded WAV byte string (standard base64, no line breaks).
    """
    samples = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_audio(audio_b64: str, expected_sample_rate: int = 16000) -> np.ndarray:
    """Decode a base64 WAV string back to a float32 numpy array.

    Inverse of _encode_audio.  Exposed for testing.

    Args:
        audio_b64: Base64-encoded WAV string.
        expected_sample_rate: Validated against the WAV header.

    Returns:
        float32 mono array normalised to [-1, 1].

    Raises:
        ValueError: If the WAV header does not match expected parameters.
    """
    raw = base64.b64decode(audio_b64)
    buf = io.BytesIO(raw)
    with wave.open(buf, "rb") as wf:
        if wf.getframerate() != expected_sample_rate:
            raise ValueError(
                f"Expected {expected_sample_rate} Hz WAV, got {wf.getframerate()} Hz"
            )
        frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return samples / 32767.0
