"""
Domain intent classification.

Maps a ContextEvent transcript to a coarse domain string using keyword
matching.  Falls back to ``"general"`` when no keywords match.

Keyword matching is case-insensitive and checks whether any keyword
appears as a substring of the transcript.

Domain → keywords
-----------------
reminder     — school, appointment, deadline, pickup, meeting
home_control — lights, thermostat, lock, door, alarm
greeting     — good morning, hello, hey heimdall
status       — what time, weather, news, score
safety       — smoke, fire, leak, motion, package
general      — (fallback)
"""

from __future__ import annotations

from nornir.models import ContextEvent

DOMAIN_INTENTS: dict[str, list[str]] = {
    "reminder": ["school", "appointment", "deadline", "pickup", "meeting"],
    "home_control": ["lights", "thermostat", "lock", "door", "alarm"],
    "greeting": ["good morning", "hello", "hey heimdall"],
    "status": ["what time", "weather", "news", "score"],
    "safety": ["smoke", "fire", "leak", "motion", "package"],
}

DOMAIN_GENERAL = "general"


def classify(event: ContextEvent) -> str:
    """Classify a ContextEvent into a domain intent.

    Scans the transcript (lowercased) for any keyword from
    ``DOMAIN_INTENTS``.  Returns the first matching domain.

    Args:
        event: Incoming context event.

    Returns:
        Domain string — one of the keys in ``DOMAIN_INTENTS`` or
        ``"general"`` if no keywords match.
    """
    text = event.transcript.lower()
    for domain, keywords in DOMAIN_INTENTS.items():
        for kw in keywords:
            if kw in text:
                return domain
    return DOMAIN_GENERAL
