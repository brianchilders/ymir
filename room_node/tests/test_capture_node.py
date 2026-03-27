"""Tests for room_node.capture_node — the Pi 4 capture-only entry point.

Verifies the run() coroutine behaves correctly without requiring real hardware.
All external deps (AudioCapture, DOAReader, PayloadSender) are mocked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from room_node.config import RoomNodeConfig

SAMPLE_RATE = 16000


def make_audio(duration_s: float = 1.0) -> np.ndarray:
    n = int(duration_s * SAMPLE_RATE)
    return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path) -> RoomNodeConfig:
    return RoomNodeConfig(
        room_name="bedroom",
        node_profile="capture",
        blackmagic_url="http://blackmagic.test:8001",
        sample_rate=SAMPLE_RATE,
        device_index=0,
        vad_threshold=0.5,
        vad_min_silence_ms=500,
        vad_speech_pad_ms=100,
        max_utterance_s=30,
        http_max_retries=1,
        http_retry_backoff_s=0.0,
        hailo_enabled=False,
        hailo_whisper_hef="./models/dummy.hef",
        hailo_emotion_hef="./models/dummy.hef",
        whisper_fallback_model="tiny",
        log_level="WARNING",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCaptureNodeRun:
    @pytest.mark.asyncio
    async def test_sends_one_utterance(self, config):
        """run() ships each utterance produced by AudioCapture."""
        utterance = make_audio(1.0)

        mock_capture = MagicMock()
        mock_capture.iter_utterances.return_value = iter([utterance])
        mock_capture.stop = MagicMock()

        mock_doa = MagicMock()
        mock_doa.read.return_value = 90

        mock_sender = MagicMock()
        mock_sender.send = AsyncMock(return_value={"ok": True})

        with (
            patch("room_node.capture_node.AudioCapture", return_value=mock_capture),
            patch("room_node.capture_node.DOAReader", return_value=mock_doa),
            patch("room_node.capture_node.PayloadSender", return_value=mock_sender),
        ):
            from room_node.capture_node import run

            # Run with a short timeout since iter_utterances is finite
            await asyncio.wait_for(run(config), timeout=5.0)

        mock_sender.send.assert_called_once()
        call_kwargs = mock_sender.send.call_args
        assert np.array_equal(call_kwargs.kwargs["audio"], utterance)
        assert call_kwargs.kwargs["doa"] == 90

    @pytest.mark.asyncio
    async def test_sender_created_with_capture_profile(self, config):
        """PayloadSender must be instantiated with node_profile='capture'."""
        mock_capture = MagicMock()
        mock_capture.iter_utterances.return_value = iter([])
        mock_capture.stop = MagicMock()

        mock_doa = MagicMock()
        mock_sender = MagicMock()
        mock_sender.send = AsyncMock(return_value=None)

        with (
            patch("room_node.capture_node.AudioCapture", return_value=mock_capture),
            patch("room_node.capture_node.DOAReader", return_value=mock_doa),
            patch("room_node.capture_node.PayloadSender", return_value=mock_sender) as MockSender,
        ):
            from room_node.capture_node import run

            await asyncio.wait_for(run(config), timeout=5.0)

        init_kwargs = MockSender.call_args.kwargs
        assert init_kwargs["node_profile"] == "capture"
        assert init_kwargs["room_name"] == "bedroom"

    @pytest.mark.asyncio
    async def test_no_hailo_inference_imported(self):
        """capture_node must not import hailo_inference (would pull heavy deps)."""
        import sys

        # Ensure the module isn't cached from a previous import
        sys.modules.pop("room_node.capture_node", None)

        hailo_before = set(k for k in sys.modules if "hailo" in k)
        import room_node.capture_node  # noqa: F401
        hailo_after = set(k for k in sys.modules if "hailo" in k)

        assert hailo_after == hailo_before, (
            "capture_node must not import hailo_inference: "
            f"new hailo modules loaded: {hailo_after - hailo_before}"
        )
