"""
Capture node entry point.

Runs on a Pi 4 (no AI accelerator).  Performs VAD + DOA only — all inference
(Whisper, emotion, resemblyzer) runs on blackmagic.  Every utterance is shipped
as a raw audio clip.

Designed to run as a systemd service::

    ExecStart=/usr/bin/python3 /opt/heimdall/capture_node.py
    Environment=ROOM_NAME=living_room
    Environment=PIPELINE_URL=http://blackmagic.lan:8001
    Environment=NODE_PROFILE=capture

Graceful shutdown is handled via SIGTERM (sent by systemd on stop).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

import httpx

from room_node.capture import AudioCapture
from room_node.config import RoomNodeConfig
from room_node.doa import DOAReader
from room_node.sender import PayloadSender

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


async def _check_connectivity(config: RoomNodeConfig) -> bool:
    """Check connectivity to the pipeline worker before starting.

    Logs the result for each service and returns False if the pipeline
    worker is unreachable (audio would just queue up with nowhere to go).

    Args:
        config: Room node configuration.

    Returns:
        True if pipeline worker is reachable, False otherwise.
    """
    ok = True
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Pipeline worker (required)
        try:
            resp = await client.get(f"{config.blackmagic_url}/health")
            data = resp.json()
            logger.info(
                "Pipeline worker reachable — %s (voiceprints cached: %s)",
                config.blackmagic_url,
                data.get("voiceprints_cached", "?"),
            )
        except Exception as exc:
            logger.error("Pipeline worker UNREACHABLE at %s — %s", config.blackmagic_url, exc)
            ok = False

    return ok


async def run(config: RoomNodeConfig) -> None:
    """Main async loop: capture → ship raw audio.

    No inference runs locally.  Every utterance is encoded and dispatched
    immediately to the pipeline worker.

    Args:
        config: Room node configuration (node_profile must be 'capture').
    """
    logger.info(
        "Capture node starting — room=%s pipeline=%s",
        config.room_name,
        config.blackmagic_url,
    )

    # Connectivity check
    reachable = await _check_connectivity(config)
    if not reachable:
        logger.warning("Pipeline worker unreachable — will queue payloads and retry on send")

    doa_reader = DOAReader()
    sender = PayloadSender(
        blackmagic_url=config.blackmagic_url,
        room_name=config.room_name,
        node_profile="capture",
        max_retries=config.http_max_retries,
        retry_backoff_s=config.http_retry_backoff_s,
        sample_rate=config.sample_rate,
        queue_maxsize=config.offline_queue_maxsize,
    )
    capture = AudioCapture(
        device_index=config.device_index,
        sample_rate=config.sample_rate,
        vad_threshold=config.vad_threshold,
        vad_min_silence_ms=config.vad_min_silence_ms,
        vad_speech_pad_ms=config.vad_speech_pad_ms,
        max_utterance_s=config.max_utterance_s,
    )

    loop = asyncio.get_event_loop()
    utterance_queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()

    def _handle_shutdown(*_: object) -> None:
        logger.info("Shutting down...")
        capture.stop()
        loop.call_soon_threadsafe(utterance_queue.put_nowait, None)
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    def _capture_loop() -> None:
        """Blocking capture loop — runs in a thread to avoid blocking the event loop."""
        for utterance in capture.iter_utterances():
            loop.call_soon_threadsafe(utterance_queue.put_nowait, utterance)
        loop.call_soon_threadsafe(utterance_queue.put_nowait, None)  # sentinel on natural exit

    async def _ship_loop() -> None:
        """Async sender loop — drains utterances from the queue and dispatches them."""
        while True:
            utterance = await utterance_queue.get()
            if utterance is None:
                break
            doa = doa_reader.read()
            await sender.send(audio=utterance, doa=doa)

    ship_task = asyncio.create_task(_ship_loop())
    executor_future = loop.run_in_executor(None, _capture_loop)

    try:
        await asyncio.wait_for(asyncio.shield(executor_future), timeout=None)
    except asyncio.CancelledError:
        pass

    await ship_task
    logger.info("Capture node shut down cleanly")
    # Force exit to unblock any remaining executor threads
    os._exit(0)


def main() -> None:
    """Entry point for the capture node service."""
    config = RoomNodeConfig()
    _configure_logging(config.log_level)
    try:
        asyncio.run(run(config))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
