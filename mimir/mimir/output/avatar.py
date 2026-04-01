"""
Avatar output via OpenHome relay.

POSTs spoken text to the Gná relay server at
``http://{RELAY_HOST}:{RELAY_PORT}/relay``.  The relay drives the
HeadTTS lip-sync animation on the OpenHome DevKit.

Failures are non-fatal — the caller falls back to TTS.
"""

from __future__ import annotations

import logging

import httpx

from mimir.config import MimirConfig

logger = logging.getLogger(__name__)


async def deliver_avatar(
    text: str,
    config: MimirConfig,
    client: httpx.AsyncClient,
) -> bool:
    """POST spoken text to the avatar relay.

    Args:
        text: Text for the avatar to speak and lip-sync.
        config: Mimir config (relay URL).
        client: Shared async HTTP client.

    Returns:
        True if the relay accepted the request (2xx), False otherwise.
    """
    try:
        resp = await client.post(
            config.relay_url,
            json={"text": text},
            timeout=5.0,
        )
        resp.raise_for_status()
        logger.debug("Avatar relay accepted: %r", text[:60])
        return True
    except httpx.HTTPError as exc:
        logger.warning("Avatar relay failed (%s) — will use TTS fallback", exc)
        return False
