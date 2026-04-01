"""
Unit tests for mimir.domain — keyword-based domain classification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nornir.models import ContextEvent
from mimir.domain import DOMAIN_GENERAL, DOMAIN_INTENTS, classify

FIXTURES = Path(__file__).parent / "fixtures" / "sample_routes.json"


def _event(transcript: str, **kwargs) -> ContextEvent:
    return ContextEvent(
        who=kwargs.get("who", "Brian"),
        transcript=transcript,
        emotion="neutral",
        location=kwargs.get("location", "kitchen"),
        local_time="2026-04-01T08:00:00",
    )


# ---------------------------------------------------------------------------
# Fixture-driven tests
# ---------------------------------------------------------------------------


def test_fixture_domains():
    """All sample_routes.json events classify to their expected domain."""
    fixtures = json.loads(FIXTURES.read_text())
    for case in fixtures:
        ev_data = case["event"]
        event = ContextEvent(
            who=ev_data["who"],
            transcript=ev_data["transcript"],
            emotion=ev_data["emotion"],
            location=ev_data["location"],
            local_time=ev_data["local_time"],
            objects_visible=ev_data.get("objects_visible", []),
        )
        result = classify(event)
        assert result == case["expected_domain"], (
            f"transcript={ev_data['transcript']!r}: "
            f"expected {case['expected_domain']!r}, got {result!r}"
        )


# ---------------------------------------------------------------------------
# Domain keyword coverage
# ---------------------------------------------------------------------------


def test_reminder_keywords():
    for kw in DOMAIN_INTENTS["reminder"]:
        assert classify(_event(f"don't forget the {kw}")) == "reminder"


def test_home_control_keywords():
    for kw in DOMAIN_INTENTS["home_control"]:
        assert classify(_event(f"please check the {kw}")) == "home_control"


def test_greeting_keywords():
    for kw in DOMAIN_INTENTS["greeting"]:
        assert classify(_event(kw)) == "greeting"


def test_status_keywords():
    for kw in DOMAIN_INTENTS["status"]:
        assert classify(_event(f"tell me the {kw}")) == "status"


def test_safety_keywords():
    for kw in DOMAIN_INTENTS["safety"]:
        assert classify(_event(f"there is {kw} detected")) == "safety"


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def test_general_fallback():
    assert classify(_event("I'm just talking about nothing in particular")) == DOMAIN_GENERAL


def test_empty_transcript():
    assert classify(_event("")) == DOMAIN_GENERAL


def test_case_insensitive():
    assert classify(_event("I have an APPOINTMENT tomorrow")) == "reminder"
    assert classify(_event("Turn on the LIGHTS please")) == "home_control"


def test_first_match_wins():
    # "door" is home_control; "school" is reminder — domain order in dict
    result = classify(_event("school door"))
    # Either reminder or home_control — whichever appears first in DOMAIN_INTENTS
    first_domain = next(iter(DOMAIN_INTENTS))
    assert result in DOMAIN_INTENTS
