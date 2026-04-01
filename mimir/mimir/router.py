"""
Mimir intent router.

Core pipeline:
  1. classify(event) → domain
  2. is_silenced(domain) → bail early if True
  3. memories empty → bail early
  4. _build_prompt(event, memories, domain) → (system, user)
  5. _generate(system, user, config, client) → raw LLM text
  6. _is_silent(raw_text) → bail if SILENT
  7. deliver to output (avatar or TTS)
  8. write_followup to Muninn
  9. record_speech in cooldown
 10. return RoutingResult

``route()`` returns ``None`` for all silent paths so the caller knows
nothing was spoken — the HTTP layer maps this to ``{ spoken: false }``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import httpx

from nornir.models import ContextEvent, RoutingResult, ScoredMemory
from mimir.config import MimirConfig
from mimir.cooldown import CooldownState
from mimir.domain import classify
from mimir.followup import write_followup
from mimir.output.avatar import deliver_avatar
from mimir.output.tts import deliver_tts

logger = logging.getLogger(__name__)

# Sentinel that signals "don't speak"
_SILENT_SENTINEL = "SILENT"


def _build_prompt(
    event: ContextEvent,
    memories: list[ScoredMemory],
    domain: str,
) -> tuple[str, str]:
    """Build the system and user prompts for the LLM.

    Args:
        event: Current context event.
        memories: Ranked memories from Verdandi.
        domain: Classified domain string.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system = (
        "You are Mimir, the voice of the Heimdall home intelligence system.\n"
        "You have been given a ranked list of relevant memories for this exact moment.\n"
        "Rules:\n"
        "- If something is urgent or timely, speak up briefly and naturally (1-2 sentences max).\n"
        "- Address the person by name.\n"
        "- If nothing is urgent or relevant, respond with only the word: SILENT\n"
        "- Never list all memories. Pick at most one thing to mention.\n"
        "- Sound like a helpful presence, not a notification system.\n"
        f"Domain context: {domain}"
    )

    memory_lines = []
    for i, mem in enumerate(memories, 1):
        memory_lines.append(
            f"{i}. [{mem.meta.get('tier', 'memory')}] (score={mem.score:.2f}) {mem.content}"
        )
    formatted_memories = "\n".join(memory_lines) if memory_lines else "(none)"

    user = (
        f"Current context:\n"
        f"  Who: {event.who}\n"
        f"  Location: {event.location}\n"
        f"  Said: {event.transcript}\n"
        f"  Time: {event.local_time}\n"
        f"  Mood: {event.emotion}\n\n"
        f"Relevant memories (score = relevance + recency + urgency):\n"
        f"{formatted_memories}\n\n"
        f"Should you say something? If yes, spoken text only. If no: SILENT"
    )

    return system, user


async def _generate(
    system: str,
    user: str,
    config: MimirConfig,
    client: httpx.AsyncClient,
) -> str:
    """Call Ollama /api/generate and return the raw response text.

    Uses ``stream=False`` for simplicity.  The SILENT sentinel is
    included in the stop list so the model never accidentally speaks it.

    Args:
        system: System prompt.
        user: User prompt.
        config: Mimir config (model, temperature, max_tokens).
        client: Shared async HTTP client.

    Returns:
        Raw stripped text from the LLM.  May be empty if SILENT triggered.

    Raises:
        httpx.HTTPStatusError: On non-2xx from Ollama.
        httpx.TimeoutException: If Ollama exceeds timeout.
    """
    payload = {
        "model": config.mimir_llm_model,
        "prompt": user,
        "system": system,
        "stream": False,
        "options": {
            "temperature": config.mimir_llm_temperature,
            "num_predict": config.mimir_llm_max_tokens,
            "stop": ["\n\n", _SILENT_SENTINEL],
        },
    }
    resp = await client.post(
        f"{config.ollama_url}/api/generate",
        json=payload,
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _is_silent(text: str) -> bool:
    """Return True if the LLM text represents a SILENT decision.

    Covers two cases:
    - Empty / whitespace-only string (SILENT was in stop list — never emitted)
    - Text starts with "SILENT" (stop list missed it)

    Args:
        text: Raw LLM output (may contain leading/trailing whitespace).

    Returns:
        True if Mimir should stay silent.
    """
    stripped = text.strip()
    if not stripped:
        return True
    return stripped.upper().startswith(_SILENT_SENTINEL)


async def route(
    event: ContextEvent,
    memories: list[ScoredMemory],
    config: MimirConfig,
    cooldown: CooldownState,
    client: httpx.AsyncClient,
    use_avatar: bool = True,
) -> Optional[RoutingResult]:
    """Main routing pipeline.  Returns None for all silent paths.

    Args:
        event: Incoming context event.
        memories: Scored memories from Verdandi (may be empty).
        config: Mimir configuration.
        cooldown: Shared cooldown state.
        client: Shared async HTTP client.
        use_avatar: Whether to attempt avatar delivery first.

    Returns:
        RoutingResult if Mimir spoke, None if silent.
    """
    t_start = time.monotonic()

    domain = classify(event)

    # --- Silence rule: cooldown ---
    if cooldown.is_silenced(domain):
        logger.debug("Silenced by cooldown (domain=%s)", domain)
        return None

    # --- Silence rule: no memories ---
    if not memories:
        logger.debug("No memories above min_score for %r — staying silent", event.who)
        return None

    # --- Build prompt + generate ---
    system, user = _build_prompt(event, memories, domain)
    try:
        raw_text = await _generate(system, user, config, client)
    except httpx.HTTPError as exc:
        logger.error("LLM generation failed: %s", exc)
        return None

    # --- Silence rule: LLM chose SILENT ---
    if _is_silent(raw_text):
        logger.debug("LLM returned SILENT for %r", event.who)
        return None

    spoken_text = raw_text

    # --- Deliver output ---
    output_path = "tts_fallback"
    if use_avatar and event.location in config.avatar_room_set:
        ok = await deliver_avatar(spoken_text, config, client)
        if ok:
            output_path = "avatar"

    if output_path != "avatar":
        await deliver_tts(spoken_text)

    # --- Write followup to Muninn ---
    await write_followup(event, spoken_text, config, client)

    # --- Update cooldown ---
    cooldown.record_speech(domain)

    latency_ms = int((time.monotonic() - t_start) * 1000)
    logger.info(
        "Mimir spoke to %r [domain=%s, path=%s, latency=%dms]: %r",
        event.who,
        domain,
        output_path,
        latency_ms,
        spoken_text[:80],
    )

    return RoutingResult(
        spoken_text=spoken_text,
        domain=domain,
        memories_used=[m.id for m in memories],
        output_path=output_path,
        latency_ms=latency_ms,
    )
