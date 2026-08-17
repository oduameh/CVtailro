"""Live model availability against OpenRouter's public catalog.

OpenRouter rotates its free-model lineup frequently; a model ID that worked
last month can 400 today. This module fetches the public catalog (no API key
required), caches it in-process for an hour, and serves stale data if
OpenRouter is unreachable so /api/models never breaks on a network hiccup.
"""

from __future__ import annotations

import logging
import threading
import time

import requests

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
CATALOG_TTL_SECONDS = 3600
FETCH_TIMEOUT_SECONDS = 5

_lock = threading.Lock()
_cache: dict = {"ids": None, "fetched_at": 0.0}


def get_live_model_ids() -> set[str] | None:
    """Return the set of model IDs currently live on OpenRouter.

    Returns None if the catalog has never been fetched successfully —
    callers should treat None as "availability unknown" and serve the
    full static list rather than hiding everything.
    """
    now = time.time()
    with _lock:
        if _cache["ids"] is not None and now - _cache["fetched_at"] < CATALOG_TTL_SECONDS:
            return _cache["ids"]

    try:
        response = requests.get(OPENROUTER_MODELS_URL, timeout=FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        ids = {m["id"] for m in response.json()["data"]}
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        logger.warning(f"Could not refresh OpenRouter model catalog: {e}")
        with _lock:
            return _cache["ids"]  # stale-if-error; None if never fetched

    with _lock:
        _cache["ids"] = ids
        _cache["fetched_at"] = now
    return ids


def filter_available(models: dict[str, str]) -> dict[str, str]:
    """Drop models no longer listed in OpenRouter's live catalog.

    If availability is unknown (catalog unreachable), returns the input
    unchanged. Never returns an empty dict — if filtering would remove
    everything, availability data is suspect and the input is returned.
    """
    live = get_live_model_ids()
    if not live:
        return models
    available = {name: mid for name, mid in models.items() if mid in live}
    if not available:
        logger.warning("Live catalog filtered out every model — serving static list")
        return models
    dropped = set(models.values()) - set(available.values())
    if dropped:
        logger.info(f"Hiding retired OpenRouter models: {sorted(dropped)}")
    return available
