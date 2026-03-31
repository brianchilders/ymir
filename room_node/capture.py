"""
Audio capture from the ReSpeaker USB Mic Array v2.0 with Silero VAD gating.

Architecture
------------
A sounddevice InputStream feeds raw PCM frames into a ring buffer.  Silero
VAD runs on 30ms chunks and gates on/off speech.  When speech ends (or
max_utterance_s is reached), the accumulated audio is yielded as a single
numpy array for downstream processing.

Fallback
--------
If no sounddevice device is found at DEVICE_INDEX, or if Silero VAD fails to
load (e.g. torch not installed), the module raises ImportError at construction
time with a clear message.  In tests, the capture loop is bypassed entirely —
audio arrays are injected directly into the processing pipeline.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # minimum chunk size required by Silero VAD (32ms at 16kHz)


class AudioCapture:
    """Capture speech segments from the ReSpeaker USB mic array.

    Runs a background sounddevice stream and exposes speech segments as an
    iterator.  Silero VAD gates silence and yields only utterances.

    Usage::

        config = RoomNodeConfig()
        capture = AudioCapture(config)
        for utterance in capture.iter_utterances():
            # utterance is float32 numpy array, 16kHz mono
            process(utterance)
    """

    def __init__(
        self,
        device_index: int = 0,
        sample_rate: int = SAMPLE_RATE,
        vad_threshold: float = 0.5,
        vad_min_silence_ms: int = 500,
        vad_speech_pad_ms: int = 100,
        max_utterance_s: int = 30,
    ) -> None:
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.vad_min_silence_ms = vad_min_silence_ms
        self.vad_speech_pad_ms = vad_speech_pad_ms
        self.max_utterance_s = max_utterance_s

        self._vad_model = _load_silero_vad()
        self._q: queue.Queue = queue.Queue()
        self._stream: Optional[object] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def iter_utterances(self) -> Iterator[np.ndarray]:
        """Yield speech utterances as float32 numpy arrays.

        Blocks until audio is available.  Intended to run in the main loop.
        Does not return unless explicitly stopped.

        Yields:
            float32 mono arrays at self.sample_rate.
        """
        self._start_stream()
        collector = _UtteranceCollector(
            vad_model=self._vad_model,
            sample_rate=self.sample_rate,
            threshold=self.vad_threshold,
            min_silence_ms=self.vad_min_silence_ms,
            speech_pad_ms=self.vad_speech_pad_ms,
            max_utterance_s=self.max_utterance_s,
        )
        try:
            while True:
                chunk = self._q.get()
                if chunk is None:  # sentinel — stop() was called
                    break
                utterance = collector.feed(chunk)
                if utterance is not None:
                    yield utterance
        finally:
            self._stop_stream()

    def stop(self) -> None:
        """Signal the capture stream to stop."""
        self._stop_stream()
        self._q.put(None)  # unblock iter_utterances if blocked on queue.get()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _start_stream(self) -> None:
        """Open the sounddevice InputStream in a background thread."""
        import sounddevice as sd

        def _callback(indata: np.ndarray, frames: int, time, status) -> None:
            if status:
                logger.debug("sounddevice status: %s", status)
            # indata shape: (frames, channels) — take channel 0 (beamformed)
            self._q.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=CHUNK_SAMPLES,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()
        logger.info(
            "Audio stream started — device=%d rate=%d chunk=%dms",
            self.device_index,
            self.sample_rate,
            30,
        )

    def _stop_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# ---------------------------------------------------------------------------
# VAD utterance collector
# ---------------------------------------------------------------------------


class _UtteranceCollector:
    """State machine that accumulates chunks and emits complete utterances.

    States: SILENCE → SPEECH → SILENCE
    Transitions governed by Silero VAD probability and silence duration.
    """

    def __init__(
        self,
        vad_model: object,
        sample_rate: int,
        threshold: float,
        min_silence_ms: int,
        speech_pad_ms: int,
        max_utterance_s: int,
    ) -> None:
        self._model = vad_model
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_silence_chunks = max(1, min_silence_ms // 30)
        self._speech_pad_chunks = max(0, speech_pad_ms // 30)
        self._max_chunks = int(max_utterance_s * 1000 / 30)

        self._in_speech = False
        self._speech_buffer: list[np.ndarray] = []
        self._silence_count = 0
        self._pad_buffer: list[np.ndarray] = []  # pre-speech padding

    def feed(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process one 30ms chunk.

        Args:
            chunk: float32 mono array of ~480 samples.

        Returns:
            Complete utterance array if one just ended, else None.
        """
        try:
            import torch
            chunk_input = torch.from_numpy(chunk)
        except ImportError:
            chunk_input = chunk  # numpy array works fine with a mocked model in tests
        prob = float(self._model(chunk_input, self._sample_rate).item())
        is_speech = prob >= self._threshold

        if not self._in_speech:
            # Maintain a rolling pad buffer of pre-speech audio
            self._pad_buffer.append(chunk)
            if len(self._pad_buffer) > self._speech_pad_chunks + 1:
                self._pad_buffer.pop(0)

            if is_speech:
                self._in_speech = True
                self._speech_buffer = list(self._pad_buffer)
                self._pad_buffer = []
                self._silence_count = 0
                logger.debug("VAD: speech start")
        else:
            self._speech_buffer.append(chunk)

            if not is_speech:
                self._silence_count += 1
            else:
                self._silence_count = 0

            # End utterance on sufficient silence or hard length cap
            timed_out = len(self._speech_buffer) >= self._max_chunks
            silent_long_enough = self._silence_count >= self._min_silence_chunks

            if timed_out or silent_long_enough:
                utterance = np.concatenate(self._speech_buffer)
                self._in_speech = False
                self._speech_buffer = []
                self._silence_count = 0
                logger.debug(
                    "VAD: speech end — %.2fs %s",
                    len(utterance) / self._sample_rate,
                    "(timeout)" if timed_out else "",
                )
                return utterance

        return None


# ---------------------------------------------------------------------------
# Silero VAD loader
# ---------------------------------------------------------------------------


def _load_silero_vad() -> object:
    """Load and return the Silero VAD model.

    Raises:
        ImportError: If torch or silero-vad is not installed.
        RuntimeError: If the model fails to load.
    """
    try:
        from silero_vad import load_silero_vad
        model = load_silero_vad()
        logger.info("Silero VAD loaded")
        return model
    except ImportError:
        # Try the older torch.hub path
        try:
            import torch
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                verbose=False,
            )
            logger.info("Silero VAD loaded via torch.hub")
            return model
        except Exception as exc:
            raise ImportError(
                "Failed to load Silero VAD. "
                "Install with: pip install silero-vad  (or torch + torchaudio)"
            ) from exc


def iter_utterances_from_array(
    audio: np.ndarray,
    vad_model: object,
    sample_rate: int = SAMPLE_RATE,
    threshold: float = 0.5,
    min_silence_ms: int = 500,
    speech_pad_ms: int = 100,
    max_utterance_s: int = 30,
) -> Iterator[np.ndarray]:
    """Process a pre-loaded audio array through VAD and yield utterances.

    This function is the testable equivalent of AudioCapture.iter_utterances().
    It takes a numpy array instead of a live microphone, making it suitable
    for unit tests and offline batch processing.

    Args:
        audio: float32 mono array at sample_rate.
        vad_model: Loaded Silero VAD model.
        sample_rate: Expected sample rate.
        threshold: VAD speech probability threshold.
        min_silence_ms: Minimum silence duration to end an utterance.
        speech_pad_ms: Audio padding prepended to each utterance.
        max_utterance_s: Hard maximum utterance length.

    Yields:
        float32 utterance arrays.
    """
    collector = _UtteranceCollector(
        vad_model=vad_model,
        sample_rate=sample_rate,
        threshold=threshold,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        max_utterance_s=max_utterance_s,
    )
    for i in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[i : i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            # Pad last chunk to full size
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        result = collector.feed(chunk)
        if result is not None:
            yield result

    # Flush any trailing speech when the array ends
    if collector._in_speech and collector._speech_buffer:
        yield np.concatenate(collector._speech_buffer)
