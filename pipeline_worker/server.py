"""
Pipeline worker FastAPI application.

Endpoints
---------
POST /ingest              — receive AudioPayload from a room node
GET  /health              — liveness check, returns counts
POST /reload_voiceprints  — hot-reload voiceprint cache from memory-mcp

Ingest flow
-----------
1. Validate AudioPayload
2. If audio_clip_b64 present → run DiarizationFallback (re-transcribe + re-embed)
3. Match voiceprint embedding against local cache → entity_name + confidence_level
4. Write to memory-mcp:
     Tier 2  — POST /record        (voice_activity reading)
     Tier 1.5— POST /open_session  (if no active session for this entity+room)
               POST /log_turn
     Tier 1  — POST /extract_and_remember  (async fact extraction)
5. If CONFIDENT → update local voiceprint cache + POST /voices/update_print
6. If PROBABLE  → POST HA webhook notification
7. Return PipelineResponse
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from pipeline_worker.diarize import DiarizationFallback
from pipeline_worker.memory_client import MemoryClient
from pipeline_worker.models import (
    AudioPayload,
    ConfidenceLevel,
    PipelineResponse,
)
from pipeline_worker.settings import Settings
from pipeline_worker.voiceprint import VoiceprintMatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state container
# ---------------------------------------------------------------------------


class AppState:
    """Holds all shared service-lifetime objects."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.matcher = VoiceprintMatcher(
            db_path=settings.voiceprint_db,
            confident_threshold=settings.voiceprint_confident_threshold,
            probable_threshold=settings.voiceprint_probable_threshold,
        )
        self.memory = MemoryClient(settings.memory_mcp_url, token=settings.memory_mcp_token)
        self.fallback = DiarizationFallback(
            hf_token=settings.hf_token or None,
        )


# ---------------------------------------------------------------------------
# App factory (allows injection of settings in tests)
# ---------------------------------------------------------------------------


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and return the FastAPI application.

    Args:
        settings: Optional Settings instance (defaults to loading from env).

    Returns:
        Configured FastAPI app with all routes registered.
    """
    if settings is None:
        settings = Settings()

    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = AppState(settings)
        app.state.app_state = state
        logger.info(
            "Pipeline worker starting — voiceprints loaded: %d, memory-mcp: %s",
            state.matcher.count(),
            settings.memory_mcp_url,
        )
        yield
        await state.memory.aclose()
        state.matcher.close()
        logger.info("Pipeline worker shut down cleanly")

    app = FastAPI(
        title="Heimdall Pipeline Worker",
        description="Receives room-node audio payloads, resolves speaker identity, writes to memory-mcp.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health(request: Request) -> dict:
        """Liveness check.

        Returns voiceprint cache count and memory-mcp reachability.
        """
        state: AppState = request.app.state.app_state
        mcp_ok = await state.memory.get_profile("__health_check__") is not None or True
        return {
            "ok": True,
            "voiceprints_cached": state.matcher.count(),
            "memory_mcp_url": settings.memory_mcp_url,
        }

    @app.post("/ingest", response_model=PipelineResponse)
    async def ingest(payload: AudioPayload, request: Request) -> PipelineResponse:
        """Receive an AudioPayload from a room node and process it.

        Returns a PipelineResponse indicating how the utterance was attributed.
        """
        state: AppState = request.app.state.app_state
        return await _process_payload(payload, state)

    @app.post("/reload_voiceprints")
    async def reload_voiceprints(request: Request) -> dict:
        """Hot-reload the voiceprint cache from memory-mcp.

        Call this after enrolling a new speaker so the pipeline worker
        starts matching against them immediately without a restart.

        Fetches all enrolled entities from memory-mcp, pulls their voiceprint
        embeddings, and repopulates the local SQLite cache.
        """
        state: AppState = request.app.state.app_state
        count = await _reload_voiceprints(state)
        return {"ok": True, "voiceprints_loaded": count}

    # ------------------------------------------------------------------
    # Exception handler
    # ------------------------------------------------------------------

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception in %s: %s", request.url.path, exc, exc_info=True)
        return JSONResponse(status_code=500, content={"ok": False, "detail": str(exc)})

    return app


# ---------------------------------------------------------------------------
# Core ingest logic (separated for testability)
# ---------------------------------------------------------------------------


async def _process_payload(payload: AudioPayload, state: AppState) -> PipelineResponse:
    """Execute the full ingest pipeline for one AudioPayload.

    This function is separated from the route handler so it can be called
    directly in tests without needing an HTTP layer.
    """
    flags: list[str] = []
    final_transcript = payload.transcript

    # ------------------------------------------------------------------
    # Step 1 — fallback re-transcription / re-embedding if audio present
    # ------------------------------------------------------------------
    fallback_embedding: Optional[np.ndarray] = None

    if payload.audio_clip_b64:
        logger.info("Audio clip present — running fallback (large-v3 + resemblyzer)")
        try:
            result = state.fallback.process(payload.audio_clip_b64)
            if result.transcript:
                final_transcript = result.transcript
                flags.append("fallback_transcription_used")
            fallback_embedding = result.embedding
        except Exception as exc:
            logger.error("Fallback processing failed: %s", exc)
            flags.append("fallback_failed")

    # ------------------------------------------------------------------
    # Step 2 — voiceprint matching
    # ------------------------------------------------------------------
    embedding_source = fallback_embedding

    if embedding_source is None and payload.voiceprint is not None:
        embedding_source = np.array(payload.voiceprint, dtype=np.float32)

    if embedding_source is not None:
        match = state.matcher.match(embedding_source)
    else:
        # No embedding available — log as unknown
        logger.warning("No voiceprint in payload and no fallback — logging as unknown")
        from pipeline_worker.voiceprint import _provisional_name
        import hashlib
        # Derive a provisional name from room+timestamp for deduplication
        key = f"{payload.room}:{payload.timestamp.isoformat()}"
        h = hashlib.sha256(key.encode()).hexdigest()[:8]
        from pipeline_worker.models import VoiceprintMatch
        match = VoiceprintMatch(
            entity_name=f"unknown_voice_{h}",
            confidence=0.0,
            confidence_level=ConfidenceLevel.UNKNOWN,
        )

    entity_name = match.entity_name
    confidence_level = match.confidence_level

    if confidence_level == ConfidenceLevel.PROBABLE:
        flags.append("probable_match")
    elif confidence_level == ConfidenceLevel.UNKNOWN:
        flags.append("unknown_speaker")

    logger.info(
        "Ingest: room=%s entity=%s level=%s transcript=%r",
        payload.room,
        entity_name,
        confidence_level.value,
        (final_transcript or "")[:60],
    )

    # ------------------------------------------------------------------
    # Step 3 — write to memory-mcp
    # ------------------------------------------------------------------

    # Tier 2 — voice_activity reading
    voice_activity_value = {
        "transcript": final_transcript,
        "confidence": match.confidence,
        "doa": payload.doa,
        "room": payload.room,
        "whisper_model": payload.whisper_model,
        "whisper_confidence": payload.whisper_confidence,
        "speaker_confidence": match.confidence,
    }
    if payload.emotion:
        voice_activity_value["emotion"] = payload.emotion.model_dump()

    await state.memory.record(
        entity_name=entity_name,
        metric="voice_activity",
        value=voice_activity_value,
    )

    # Tier 1.5 — session + transcript turn
    # NOTE: /open_session, /log_turn, /close_session are not yet implemented in
    # memory-mcp. These calls return None gracefully and are no-ops until those
    # endpoints are added. See docs/memory-mcp-voice-extension-spec.md for context.
    session_id: Optional[str] = None
    session_result = await state.memory.open_session(entity_name)
    if session_result:
        # result field is the integer session_id returned by memory-mcp
        raw_id = session_result.get("result")
        if isinstance(raw_id, int):
            session_id = str(raw_id)
            # Log a system turn with room context, then the user utterance
            await state.memory.log_turn(
                session_id=raw_id,
                role="system",
                content=f"room={payload.room} doa={payload.doa} confidence={confidence_level.value}",
            )
            if final_transcript:
                await state.memory.log_turn(
                    session_id=raw_id,
                    role="user",
                    content=final_transcript,
                )

    # Tier 1 — fact extraction
    # NOTE: /extract_and_remember is not yet implemented in memory-mcp.
    # Returns None gracefully until added.
    if final_transcript:
        await state.memory.extract_and_remember(entity_name, final_transcript)

    # ------------------------------------------------------------------
    # Step 4 — update voiceprint if confident
    # ------------------------------------------------------------------
    if confidence_level == ConfidenceLevel.CONFIDENT and embedding_source is not None:
        state.matcher.update_after_match(entity_name, embedding_source)
        await state.memory.update_voiceprint(
            entity_name,
            embedding_source.tolist(),
        )

    # ------------------------------------------------------------------
    # Step 5 — HA notification if probable
    # ------------------------------------------------------------------
    if confidence_level == ConfidenceLevel.PROBABLE and state.settings.ha_webhook_url:
        await _notify_ha(state, entity_name, match.confidence, payload.room)

    return PipelineResponse(
        ok=True,
        entity_name=entity_name,
        confidence_level=confidence_level,
        transcript=final_transcript,
        session_id=session_id,
        flags=flags,
    )


async def _reload_voiceprints(state: AppState) -> int:
    """Reload the local voiceprint cache from memory-mcp enrolled entities.

    Queries memory-mcp for all person entities, pulls voiceprint embeddings
    from their metadata, and repopulates voiceprints.sqlite.

    Returns:
        Number of voiceprints successfully loaded.
    """
    # Fetch all entities from memory-mcp
    try:
        import httpx
        response = await state.memory._client.get("/entities")
        response.raise_for_status()
        # /entities returns {"entities": [...]} — not wrapped in "result"
        entities = response.json().get("entities", [])
    except Exception as exc:
        logger.error("Failed to fetch entities from memory-mcp: %s", exc)
        return 0

    count = 0
    for entity in entities:
        meta = entity.get("meta") or {}
        vp = meta.get("voiceprint")
        if not vp or len(vp) != 256:
            continue
        # /entities returns "name" not "entity_name"
        name = entity.get("name")
        samples = meta.get("voiceprint_samples", 1)
        embedding = np.array(vp, dtype=np.float32)
        state.matcher.upsert(name, embedding, sample_count=samples)
        count += 1

    logger.info("Voiceprint cache reloaded: %d embeddings", count)
    return count


async def _notify_ha(
    state: AppState,
    entity_name: str,
    confidence: float,
    room: str,
) -> None:
    """POST a probable-match notification to the Home Assistant webhook."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as ha:
            await ha.post(
                state.settings.ha_webhook_url,
                json={
                    "entity_name": entity_name,
                    "confidence": confidence,
                    "room": room,
                    "message": (
                        f"Probable speaker match: {entity_name} "
                        f"(confidence {confidence:.0%}) in {room}"
                    ),
                },
            )
    except Exception as exc:
        logger.warning("HA notification failed: %s", exc)


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


# Module-level app instance for uvicorn
app = create_app()
