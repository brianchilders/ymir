"""
Followup writer.

After Mimir speaks, it writes a pending followup to Muninn so the
next time that person is detected, Mimir can check whether a response
or confirmation is needed.
"""

from __future__ import annotations

import logging

import httpx

from nornir.models import ContextEvent
from mimir.config import MimirConfig

logger = logging.getLogger(__name__)


async def write_followup(
    event: ContextEvent,
    spoken_text: str,
    config: MimirConfig,
    client: httpx.AsyncClient,
) -> None:
    """POST a pending followup to Muninn.

    Failures are logged but do not propagate — a missed followup is
    not a critical error and must not silence the response.

    Args:
        event: The context event that triggered speech.
        spoken_text: What Mimir said (stored for reference).
        config: Mimir config (Muninn URL, TTL).
        client: Shared async HTTP client.
    """
    body = {
        "who": event.who,
        "spoken_text": spoken_text,
        "location": event.location,
        "ttl_hours": 4,
    }
    try:
        resp = await client.post(
            f"{config.muninn_url}/followups",
            json=body,
            timeout=5.0,
        )
        resp.raise_for_status()
        logger.debug("Followup stored for %r", event.who)
    except httpx.HTTPError as exc:
        logger.warning("Could not write followup to Muninn: %s", exc)
