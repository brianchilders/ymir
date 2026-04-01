"""Entrypoint: python -m mimir"""

from __future__ import annotations

import logging

import uvicorn

from mimir.config import MimirConfig

config = MimirConfig()

logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

uvicorn.run(
    "mimir.api.app:app",
    host="0.0.0.0",
    port=config.mimir_port,
    log_level=config.log_level.lower(),
)
