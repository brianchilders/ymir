"""
Async HTTP client for the memory-mcp API.

Wraps all memory-mcp endpoints used by the pipeline worker and enrollment
CLI into typed async methods.  All errors are caught and logged — the caller
receives None or an empty result rather than an exception, allowing the
pipeline to continue processing even when memory-mcp is temporarily
unavailable.

Endpoints covered
-----------------
Tier 2  — /record
Tier 1.5— /open_session, /log_turn, /close_session
Tier 1  — /remember, /extract_and_remember, /relate, /recall, /profile/{name}
Voice   — /voices/unknown, /voices/enroll, /voices/merge, /voices/update_print
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Default timeout for all requests (seconds).
DEFAULT_TIMEOUT = 10.0


class MemoryClient:
    """Async client for the memory-mcp HTTP API.

    Intended to be used as an async context manager::

        async with MemoryClient("http://localhost:8900") as client:
            await client.record(entity_name="Brian", metric="voice_activity", value={...})

    Or kept as a long-lived instance in the pipeline worker::

        client = MemoryClient("http://localhost:8900")
        # ... use throughout service lifetime ...
        await client.aclose()
    """

    def __init__(self, base_url: str, token: str = "", timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "MemoryClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Tier 2 — Time-series readings
    # ------------------------------------------------------------------

    async def record(
        self,
        entity_name: str,
        metric: str,
        value: Any,
        unit: Optional[str] = None,
        source: str = "audio_pipeline",
    ) -> Optional[dict]:
        """Ingest a time-series reading (Tier 2).

        Used to log voice_activity readings with transcript, emotion, DOA.

        Args:
            entity_name: The entity this reading belongs to.
            metric: Metric name, e.g. 'voice_activity'.
            value: Composite value dict or scalar.
            unit: Optional unit string.
            source: Data source tag.

        Returns:
            Parsed JSON response or None on error.
        """
        payload: dict[str, Any] = {
            "entity_name": entity_name,
            "metric": metric,
            "value": value,
            "source": source,
        }
        if unit:
            payload["unit"] = unit
        return await self._post("/record", payload)

    # ------------------------------------------------------------------
    # Tier 1.5 — Episodic memory (conversation sessions)
    # ------------------------------------------------------------------

    async def open_session(
        self,
        entity_name: str,
        entity_type: str = "person",
    ) -> Optional[dict]:
        """Start a new conversation session for an entity (Tier 1.5).

        Returns a response whose 'result' field is the integer session_id.
        Pass that id to log_turn and close_session.

        Args:
            entity_name: Speaker identity for this session.
            entity_type: Entity type (default 'person').

        Returns:
            Response dict with integer session_id in result, or None on error.
        """
        return await self._post(
            "/open_session",
            {"entity_name": entity_name, "entity_type": entity_type},
        )

    async def log_turn(
        self,
        session_id: int,
        role: str,
        content: str,
    ) -> Optional[dict]:
        """Append a transcript turn to an open session (Tier 1.5).

        Args:
            session_id: Integer ID returned by open_session.
            role: One of 'user', 'assistant', 'system'.
            content: Transcribed utterance or system annotation text.

        Returns:
            Parsed JSON response or None on error.
        """
        return await self._post(
            "/log_turn",
            {"session_id": session_id, "role": role, "content": content},
        )

    async def close_session(
        self,
        session_id: int,
        summary: Optional[str] = None,
    ) -> Optional[dict]:
        """Close a conversation session (Tier 1.5).

        Args:
            session_id: Integer ID of the session to close.
            summary: Optional summary string stored with the session.

        Returns:
            Parsed JSON response or None on error.
        """
        payload: dict[str, Any] = {"session_id": session_id}
        if summary:
            payload["summary"] = summary
        return await self._post("/close_session", payload)

    async def get_session(self, session_id: int) -> Optional[dict]:
        """Retrieve a full session transcript with all turns and summary (Tier 1.5).

        memory-mcp always returns HTTP 200.  A not-found session is indicated
        by ``result`` being a string like "No session with id=N." rather than
        a dict.

        Args:
            session_id: Integer session ID returned by open_session.

        Returns:
            Parsed response dict (with dict ``result``) or None if not found / on error.
        """
        try:
            response = await self._client.get(f"/get_session/{session_id}")
            response.raise_for_status()
            data = response.json()
            result = data.get("result", "")
            if isinstance(result, str) and result.startswith("No session with id"):
                logger.debug("Session not found: %d", session_id)
                return None
            return data
        except httpx.HTTPStatusError as exc:
            logger.error("memory-mcp GET /get_session/%d error: %s", session_id, exc)
            return None
        except httpx.RequestError as exc:
            logger.error("memory-mcp connection error (GET /get_session/%d): %s", session_id, exc)
            return None

    # ------------------------------------------------------------------
    # Tier 1 — Semantic memory
    # ------------------------------------------------------------------

    async def remember(
        self,
        entity_name: str,
        fact: str,
        category: Optional[str] = None,
        confidence: float = 1.0,
        source: str = "audio_pipeline",
    ) -> Optional[dict]:
        """Store a fact about an entity (Tier 1).

        Args:
            entity_name: Entity this fact is about.
            fact: Natural-language fact string.
            category: Optional taxonomy tag (e.g. 'preference', 'location').
            confidence: Confidence score 0–1.
            source: Data source tag.

        Returns:
            Parsed JSON response or None on error.
        """
        payload: dict[str, Any] = {
            "entity_name": entity_name,
            "fact": fact,
            "confidence": confidence,
            "source": source,
        }
        if category:
            payload["category"] = category
        return await self._post("/remember", payload)

    async def extract_and_remember(
        self,
        entity_name: str,
        text: str,
        entity_type: str = "person",
        model: Optional[str] = None,
    ) -> Optional[dict]:
        """LLM-powered fact extraction from free text, stored to Tier 1.

        Delegates to memory-mcp which calls Ollama to extract facts.
        Field names match ExtractAndRememberRequest in memory-mcp api.py.

        Args:
            entity_name: Entity the text is about / spoken by.
            text: Utterance or passage to extract facts from.
            entity_type: Entity type (default 'person').
            model: Ollama model override (default None → uses LLM_MODEL env var).

        Returns:
            Parsed JSON response with extraction summary, or None on error.
        """
        payload: dict[str, Any] = {
            "entity_name": entity_name,
            "text": text,
            "entity_type": entity_type,
        }
        if model:
            payload["model"] = model
        return await self._post("/extract_and_remember", payload)

    async def relate(
        self,
        entity_a: str,
        rel_type: str,
        entity_b: str,
    ) -> Optional[dict]:
        """Create a relationship between two entities (Tier 1).

        Field names match memory-mcp's RelateRequest model exactly.

        Args:
            entity_a: First entity name.
            rel_type: Relationship type string, e.g. 'lives_with'.
            entity_b: Second entity name.

        Returns:
            Parsed JSON response or None on error.
        """
        return await self._post(
            "/relate",
            {"entity_a": entity_a, "rel_type": rel_type, "entity_b": entity_b},
        )

    async def recall(
        self,
        query: str,
        entity_name: Optional[str] = None,
        top_k: int = 5,
    ) -> Optional[dict]:
        """Semantic memory search (Tier 1).

        Args:
            query: Natural-language search query.
            entity_name: Optional — restrict search to one entity.
            top_k: Maximum number of results to return.

        Returns:
            Parsed JSON response with matching memories, or None on error.
        """
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if entity_name:
            payload["entity_name"] = entity_name
        return await self._post("/recall", payload)

    async def get_profile(self, entity_name: str) -> Optional[dict]:
        """Retrieve the full profile for an entity (Tier 1).

        memory-mcp always returns HTTP 200.  A not-found entity is indicated
        by ``result`` being a string like "No entity named 'X'." rather than
        a dict.

        Args:
            entity_name: Entity to look up.

        Returns:
            Parsed response dict (with dict ``result``) or None if not found / on error.
        """
        try:
            response = await self._client.get(f"/profile/{entity_name}")
            response.raise_for_status()
            data = response.json()
            result = data.get("result", "")
            if isinstance(result, str) and result.startswith("No entity named"):
                logger.debug("Entity not found in memory-mcp: %s", entity_name)
                return None
            return data
        except httpx.HTTPStatusError as exc:
            logger.error("memory-mcp GET /profile/%s error: %s", entity_name, exc)
            return None
        except httpx.RequestError as exc:
            logger.error("memory-mcp connection error (GET /profile/%s): %s", entity_name, exc)
            return None

    # ------------------------------------------------------------------
    # Voice management routes (added to memory-mcp per spec)
    # ------------------------------------------------------------------

    async def list_unknown_voices(
        self,
        limit: int = 20,
        min_detections: int = 1,
    ) -> Optional[list[dict]]:
        """List all unenrolled provisional speaker entities.

        Args:
            limit: Maximum number of results.
            min_detections: Filter out entities with fewer detections.

        Returns:
            List of voice dicts or None on error.
        """
        try:
            response = await self._client.get(
                "/voices/unknown",
                params={"limit": limit, "min_detections": min_detections},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("result", [])
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.error("memory-mcp GET /voices/unknown error: %s", exc)
            return None

    async def enroll_voice(
        self,
        entity_name: str,
        new_name: str,
        display_name: Optional[str] = None,
    ) -> Optional[dict]:
        """Rename a provisional entity to a real person name.

        Args:
            entity_name: Current provisional name, e.g. 'unknown_voice_a3f2'.
            new_name: Real name to assign, e.g. 'Brian'.
            display_name: Optional human-readable full name.

        Returns:
            Enrollment result dict or None on error.
        """
        payload: dict[str, Any] = {
            "entity_name": entity_name,
            "new_name": new_name,
        }
        if display_name:
            payload["display_name"] = display_name
        return await self._post("/voices/enroll", payload)

    async def merge_voices(
        self,
        source_name: str,
        target_name: str,
    ) -> Optional[dict]:
        """Merge a provisional entity into an enrolled entity.

        Transfers all memories, readings, and relations from source to target,
        averages voiceprint embeddings, then deletes the source entity.

        Args:
            source_name: Provisional entity to merge from.
            target_name: Enrolled entity to merge into.

        Returns:
            Merge result dict or None on error.
        """
        return await self._post(
            "/voices/merge",
            {"source_name": source_name, "target_name": target_name},
        )

    async def update_voiceprint(
        self,
        entity_name: str,
        embedding: list[float],
        weight: float = 0.1,
    ) -> Optional[dict]:
        """Update the canonical voiceprint embedding in memory-mcp.

        Called after each confident match to keep the canonical embedding in
        sync with the local VoiceprintMatcher cache.

        Args:
            entity_name: Entity to update.
            embedding: 256-dim float list (resemblyzer output).
            weight: Contribution of the new sample (default 0.1).

        Returns:
            Update result dict or None on error.
        """
        return await self._post(
            "/voices/update_print",
            {"entity_name": entity_name, "embedding": embedding, "weight": weight},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post(self, path: str, payload: dict) -> Optional[dict]:
        """POST JSON payload to a memory-mcp endpoint.

        Catches all HTTP and network errors, logs them, and returns None so
        the calling pipeline can continue without crashing.

        Args:
            path: URL path, e.g. '/record'.
            payload: JSON-serialisable dict.

        Returns:
            Parsed JSON response dict or None on error.
        """
        try:
            response = await self._client.post(path, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "memory-mcp POST %s returned %d: %s",
                path,
                exc.response.status_code,
                exc.response.text[:200],
            )
            return None
        except httpx.RequestError as exc:
            logger.error("memory-mcp connection error (POST %s): %s", path, exc)
            return None
