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
import signal

from room_node.capture import AudioCapture
from room_node.config import RoomNodeConfig
from room_node.doa import DOAReader
from room_node.sender import PayloadSender


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


async def run(config: RoomNodeConfig) -> None:
    """Main async loop: capture → ship raw audio.

    No inference runs locally.  Every utterance is encoded and dispatched
    immediately to the pipeline worker.

    Args:
        config: Room node configuration (node_profile must be 'capture').
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "Capture node starting — room=%s pipeline=%s",
        config.room_name,
        config.blackmagic_url,
    )

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
    stop_event = asyncio.Event()

    def _handle_sigterm(*_: object) -> None:
        logger.info("SIGTERM received — shutting down")
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    utterance_queue: asyncio.Queue = asyncio.Queue()

    def _capture_loop() -> None:
        """Blocking capture loop — runs in a thread to avoid blocking the event loop."""
        for utterance in capture.iter_utterances():
            if stop_event.is_set():
                break
            loop.call_soon_threadsafe(utterance_queue.put_nowait, utterance)
        loop.call_soon_threadsafe(utterance_queue.put_nowait, None)  # sentinel

    async def _ship_loop() -> None:
        """Async sender loop — drains utterances from the queue and dispatches them."""
        while True:
            utterance = await utterance_queue.get()
            if utterance is None:
                break
            doa = doa_reader.read()
            await sender.send(audio=utterance, doa=doa)
        stop_event.set()

    capture_task = asyncio.create_task(_ship_loop())
    await loop.run_in_executor(None, _capture_loop)
    await stop_event.wait()
    capture.stop()
    capture_task.cancel()
    logger.info("Capture node shut down cleanly")


def main() -> None:
    """Entry point for the capture node service."""
    config = RoomNodeConfig()
    _configure_logging(config.log_level)
    try:
        asyncio.run(run(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
