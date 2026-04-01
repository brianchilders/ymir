"""
Unit tests for mimir.cooldown.CooldownState.
"""

from __future__ import annotations

import time

import pytest

from mimir.cooldown import CooldownState


def _make(cooldown_s: int = 90, greeting_s: int = 300) -> CooldownState:
    return CooldownState(cooldown_s=cooldown_s, greeting_cooldown_s=greeting_s)


# ---------------------------------------------------------------------------
# Initial state — not silenced
# ---------------------------------------------------------------------------


def test_not_silenced_initially():
    cs = _make()
    assert cs.is_silenced("general") is False
    assert cs.is_silenced("greeting") is False


def test_remaining_zero_initially():
    cs = _make()
    assert cs.remaining_seconds() == 0.0


# ---------------------------------------------------------------------------
# After speech
# ---------------------------------------------------------------------------


def test_silenced_immediately_after_speech():
    cs = _make(cooldown_s=90)
    cs.record_speech("general")
    assert cs.is_silenced("general") is True


def test_remaining_positive_after_speech():
    cs = _make(cooldown_s=90)
    cs.record_speech("general")
    remaining = cs.remaining_seconds()
    assert 88.0 < remaining <= 90.0


def test_not_silenced_after_cooldown_expires():
    cs = _make(cooldown_s=1)
    cs.record_speech("general")
    time.sleep(1.1)
    assert cs.is_silenced("general") is False


# ---------------------------------------------------------------------------
# Greeting cooldown
# ---------------------------------------------------------------------------


def test_greeting_silenced_after_greeting():
    cs = _make(cooldown_s=1, greeting_s=300)
    cs.record_speech("greeting")
    time.sleep(1.1)  # global cooldown expires
    # greeting cooldown still active
    assert cs.is_silenced("greeting") is True


def test_non_greeting_not_silenced_by_greeting_cooldown():
    cs = _make(cooldown_s=1, greeting_s=300)
    cs.record_speech("greeting")
    time.sleep(1.1)
    # "general" domain — only global cooldown applies
    assert cs.is_silenced("general") is False


def test_greeting_cooldown_expires():
    cs = _make(cooldown_s=1, greeting_s=1)
    cs.record_speech("greeting")
    time.sleep(1.2)
    assert cs.is_silenced("greeting") is False


# ---------------------------------------------------------------------------
# Status dict
# ---------------------------------------------------------------------------


def test_status_before_speech():
    cs = _make()
    s = cs.status()
    assert s["remaining_s"] == 0.0
    assert s["last_spoken"] is None
    assert s["last_greeting"] is None


def test_status_after_speech():
    cs = _make(cooldown_s=90)
    cs.record_speech("general")
    s = cs.status()
    assert s["remaining_s"] > 0
    assert s["last_spoken"] is not None
    assert s["last_greeting"] is None


def test_status_after_greeting():
    cs = _make(cooldown_s=90)
    cs.record_speech("greeting")
    s = cs.status()
    assert s["last_greeting"] is not None


# ---------------------------------------------------------------------------
# Multiple speeches reset cooldown
# ---------------------------------------------------------------------------


def test_cooldown_reset_on_second_speech():
    cs = _make(cooldown_s=2)
    cs.record_speech("general")
    time.sleep(1.0)
    # Still silenced
    assert cs.is_silenced("general") is True
    cs.record_speech("general")
    # Cooldown reset — full 2s again
    remaining = cs.remaining_seconds()
    assert remaining > 1.5
