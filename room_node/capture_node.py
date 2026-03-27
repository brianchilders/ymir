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

    async def _capture_and_ship() -> None:
        for utterance in capture.iter_utterances():
            if stop_event.is_set():
                break

            doa = doa_reader.read()
            await sender.send(audio=utterance, doa=doa)
        # Loop exited (iterator exhausted or stop requested) — unblock run()
        stop_event.set()

    capture_task = asyncio.create_task(_capture_and_ship())
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
