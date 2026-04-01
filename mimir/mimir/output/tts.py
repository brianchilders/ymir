"""
TTS fallback using pyttsx3.

pyttsx3 is synchronous; we run it in a thread-pool executor to avoid
blocking the asyncio event loop.  The import is optional — if pyttsx3
is not installed, ``deliver_tts`` logs a warning and returns without
speaking (useful in CI/test environments).
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

try:
    import pyttsx3
    _HAS_PYTTSX3 = True
except ImportError:
    _HAS_PYTTSX3 = False


def _speak_sync(text: str) -> None:
    """Synchronous TTS call — runs in a thread executor.

    Args:
        text: Text to speak aloud.
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


async def deliver_tts(text: str) -> None:
    """Speak text via pyttsx3 in a thread executor.

    If pyttsx3 is not installed, logs a warning and returns immediately.
    This allows the service to run on machines without audio output
    (e.g. CI, headless servers) without crashing.

    Args:
        text: Text to speak aloud.
    """
    if not _HAS_PYTTSX3:
        logger.warning("pyttsx3 not installed — TTS fallback unavailable. Text: %r", text[:80])
        return

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _speak_sync, text)
        logger.debug("TTS delivered: %r", text[:60])
    except Exception as exc:
        logger.error("TTS error: %s", exc)
