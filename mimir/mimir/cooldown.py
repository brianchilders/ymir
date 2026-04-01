"""
Silence cooldown state.

Tracks when Mimir last spoke and enforces two cooldown rules:
  1. Global cooldown  — silence_cooldown_seconds after any utterance.
  2. Greeting cooldown — greeting_cooldown_minutes after a greeting.

This is in-process state (not persisted across restarts).  A single
``CooldownState`` instance lives in ``app.state`` for the lifetime of
the service process.
"""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock


class CooldownState:
    """Thread-safe in-memory cooldown tracker.

    Args:
        cooldown_s: Seconds to stay silent after any utterance.
        greeting_cooldown_s: Seconds to stay silent after a greeting.
    """

    def __init__(self, cooldown_s: int, greeting_cooldown_s: int) -> None:
        self._cooldown_s = cooldown_s
        self._greeting_cooldown_s = greeting_cooldown_s
        self._last_spoken: datetime | None = None
        self._last_greeting: datetime | None = None
        self._lock = Lock()

    def is_silenced(self, domain: str) -> bool:
        """Return True if Mimir should stay silent right now.

        Silenced when:
        - Within ``cooldown_s`` of last utterance, OR
        - Domain is ``"greeting"`` and within ``greeting_cooldown_s``
          of last greeting.

        Args:
            domain: Classified domain for the current event.

        Returns:
            True if silent, False if Mimir may speak.
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            if self._last_spoken is not None:
                elapsed = (now - self._last_spoken).total_seconds()
                if elapsed < self._cooldown_s:
                    return True
            if domain == "greeting" and self._last_greeting is not None:
                elapsed_g = (now - self._last_greeting).total_seconds()
                if elapsed_g < self._greeting_cooldown_s:
                    return True
        return False

    def record_speech(self, domain: str) -> None:
        """Record that Mimir just spoke.

        Updates last_spoken (always) and last_greeting (if greeting domain).

        Args:
            domain: Domain of the utterance just delivered.
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            self._last_spoken = now
            if domain == "greeting":
                self._last_greeting = now

    def remaining_seconds(self) -> float:
        """Seconds remaining in the global cooldown (0 if not in cooldown).

        Returns:
            Float seconds, floored at 0.
        """
        with self._lock:
            if self._last_spoken is None:
                return 0.0
            elapsed = (datetime.now(timezone.utc) - self._last_spoken).total_seconds()
            return max(0.0, self._cooldown_s - elapsed)

    def status(self) -> dict:
        """Return a snapshot of cooldown state for the /cooldown endpoint.

        Returns:
            Dict with ``remaining_s``, ``last_spoken``, ``last_greeting``.
        """
        with self._lock:
            return {
                "remaining_s": self.remaining_seconds(),
                "last_spoken": self._last_spoken.isoformat() if self._last_spoken else None,
                "last_greeting": self._last_greeting.isoformat() if self._last_greeting else None,
            }
