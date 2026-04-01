"""
Mimir FastAPI application.

Routes:
  POST /route     — receive ContextEvent, fetch memories from Verdandi, route
  GET  /cooldown  — current cooldown state
  GET  /health    — liveness + upstream connectivity
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nornir.models import ContextEvent, ScoredMemory
from mimir.config import MimirConfig
from mimir.cooldown import CooldownState
from mimir.router import route

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ContextEventRequest(BaseModel):
    """Pydantic mirror of nornir.ContextEvent for HTTP deserialization."""

    who: str
    transcript: str
    emotion: str
    location: str
    local_time: str
    speaker_confidence: float = 0.0
    doa_degrees: Optional[int] = None
    objects_visible: list[str] = []
    people_detected: list[str] = []
    activity: Optional[str] = None

    def to_event(self) -> ContextEvent:
        """Convert to the nornir ContextEvent dataclass."""
        return ContextEvent(
            who=self.who,
            transcript=self.transcript,
            emotion=self.emotion,
            location=self.location,
            local_time=self.local_time,
            speaker_confidence=self.speaker_confidence,
            doa_degrees=self.doa_degrees,
            objects_visible=self.objects_visible,
            people_detected=self.people_detected,
            activity=self.activity,
        )


class RouteRequest(BaseModel):
    """Body for POST /route."""

    event: ContextEventRequest
    top_k: int = 5
    use_avatar: bool = True


class RouteResponse(BaseModel):
    """Response from POST /route."""

    spoken: bool
    text: Optional[str] = None
    domain: str
    output_path: Optional[str] = None
    latency_ms: int


# ---------------------------------------------------------------------------
# Verdandi client — fetch recommendations inline
# ---------------------------------------------------------------------------


async def _fetch_memories(
    event: ContextEvent,
    top_k: int,
    config: MimirConfig,
    client: httpx.AsyncClient,
) -> list[ScoredMemory]:
    """Call Verdandi POST /recommend and return ScoredMemory list.

    Returns an empty list on any error so routing can still decide
    to be silent rather than crashing.

    Args:
        event: Context event to recommend for.
        top_k: Number of memories to request.
        config: Mimir config (Verdandi URL).
        client: Shared async HTTP client.

    Returns:
        List of ScoredMemory, possibly empty.
    """
    body = {
        "event": {
            "who": event.who,
            "transcript": event.transcript,
            "emotion": event.emotion,
            "location": event.location,
            "local_time": event.local_time,
            "speaker_confidence": event.speaker_confidence,
            "doa_degrees": event.doa_degrees,
            "objects_visible": event.objects_visible,
            "people_detected": event.people_detected,
            "activity": event.activity,
        },
        "top_k": top_k,
    }
    try:
        resp = await client.post(
            f"{config.verdandi_url}/recommend",
            json=body,
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            ScoredMemory(
                id=r["id"],
                content=r["content"],
                score=r["score"],
                similarity=r["similarity"],
                recency=r["recency"],
                urgency=r["urgency"],
                meta=r.get("meta", {}),
            )
            for r in data.get("recommendations", [])
        ]
    except httpx.HTTPError as exc:
        logger.warning("Verdandi unavailable: %s — routing with empty memories", exc)
        return []


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


def _make_lifespan(config: MimirConfig):
    """Return a lifespan context bound to this config."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient()
        app.state.config = config
        app.state.cooldown = CooldownState(
            cooldown_s=config.silence_cooldown_seconds,
            greeting_cooldown_s=config.greeting_cooldown_minutes * 60,
        )
        logger.info(
            "Mimir ready (verdandi=%s, ollama=%s, model=%s)",
            config.verdandi_url,
            config.ollama_url,
            config.mimir_llm_model,
        )
        yield
        await app.state.client.aclose()
        logger.info("Mimir stopped")

    return lifespan


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: MimirConfig | None = None) -> FastAPI:
    """Build and return the configured FastAPI application.

    Args:
        config: Optional MimirConfig.  Reads from env if omitted.

    Returns:
        FastAPI instance.
    """
    cfg = config or MimirConfig()

    app = FastAPI(
        title="Mimir Intent Router",
        version="0.1.0",
        description="Decides whether to speak, generates text, routes to avatar or TTS.",
        lifespan=_make_lifespan(cfg),
    )

    @app.post("/route", response_model=RouteResponse)
    async def route_event(body: RouteRequest) -> RouteResponse:
        """Receive a ContextEvent, fetch memories, decide whether to speak.

        Args:
            body: Route request with event and options.

        Returns:
            RouteResponse indicating whether Mimir spoke and what it said.

        Raises:
            HTTPException 500: On unexpected internal error.
        """
        event = body.event.to_event()
        memories = await _fetch_memories(
            event, body.top_k, cfg, app.state.client
        )

        from mimir.domain import classify
        domain = classify(event)

        result = await route(
            event=event,
            memories=memories,
            config=cfg,
            cooldown=app.state.cooldown,
            client=app.state.client,
            use_avatar=body.use_avatar,
        )

        if result is None:
            return RouteResponse(
                spoken=False,
                text=None,
                domain=domain,
                output_path=None,
                latency_ms=0,
            )

        return RouteResponse(
            spoken=True,
            text=result.spoken_text,
            domain=result.domain,
            output_path=result.output_path,
            latency_ms=result.latency_ms,
        )

    @app.get("/cooldown", tags=["ops"])
    async def cooldown_status() -> dict:
        """Return current cooldown state.

        Returns:
            Dict with remaining_s, last_spoken, last_greeting.
        """
        return app.state.cooldown.status()

    @app.get("/health", tags=["ops"])
    async def health() -> dict:
        """Liveness + upstream connectivity check.

        Returns:
            Dict with status, verdandi, ollama fields.
        """
        verdandi_ok = False
        ollama_ok = False
        verdandi_detail = "unknown"
        ollama_detail = "unknown"

        try:
            resp = await app.state.client.get(
                f"{cfg.verdandi_url}/health", timeout=3.0
            )
            verdandi_ok = resp.status_code == 200
            verdandi_detail = "connected" if verdandi_ok else f"http {resp.status_code}"
        except Exception as exc:
            verdandi_detail = str(exc)

        try:
            resp = await app.state.client.get(
                f"{cfg.ollama_url}/api/tags", timeout=3.0
            )
            ollama_ok = resp.status_code == 200
            ollama_detail = "connected" if ollama_ok else f"http {resp.status_code}"
        except Exception as exc:
            ollama_detail = str(exc)

        return {
            "status": "ok" if (verdandi_ok and ollama_ok) else "degraded",
            "verdandi": verdandi_detail,
            "ollama": ollama_detail,
            "model": cfg.mimir_llm_model,
        }

    return app


app = create_app()
