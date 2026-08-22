"""Premium model access control — single source of truth.

Free models are available to everyone. Premium (paid) OpenRouter models are
usable only when the global ``premium_models_enabled`` setting is on AND the
caller is an admin or a user an admin explicitly granted. Premium model IDs
are derived from config.RECOMMENDED_MODELS — never duplicated here.
"""

from __future__ import annotations

from config import RECOMMENDED_MODELS

PREMIUM_MODELS_ENABLED_KEY = "premium_models_enabled"


def _is_free_entry(display_name: str, model_id: str) -> bool:
    """Curated free entries carry the ':free' ID suffix and/or a '(Free)' label."""
    return model_id.endswith(":free") or "(Free)" in display_name


# Every curated model that costs credits (the paid groups in config.py).
PREMIUM_MODEL_IDS: frozenset[str] = frozenset(
    model_id
    for display_name, model_id in RECOMMENDED_MODELS.items()
    if not _is_free_entry(display_name, model_id)
)


def is_premium_model(model_id: str) -> bool:
    """Return True if the model is a paid (premium) OpenRouter model."""
    return model_id in PREMIUM_MODEL_IDS


def premium_models_enabled() -> bool:
    """Read the global kill switch. Unset defaults to enabled."""
    from app.services.admin_config import AdminConfigManager

    return (AdminConfigManager.get(PREMIUM_MODELS_ENABLED_KEY) or "true") != "false"


def can_use_premium_models(user) -> bool:
    """True only when the feature is on AND the user is an admin or granted."""
    if user is None or not premium_models_enabled():
        return False
    return bool(getattr(user, "is_admin", False) or getattr(user, "can_use_premium_models", False))


def filter_catalog_for_user(catalog: dict | list, user) -> dict | list:
    """Filter a /api/models payload for the caller.

    Annotates every model dict with ``tier`` ("free"|"premium"), drops premium
    entries the user may not use, and sets top-level ``user_can_use_premium``.
    Accepts the full response dict or a bare list of model dicts; returns the
    same shape it was given.
    """
    allowed = can_use_premium_models(user)

    def _usable(entry: dict) -> bool:
        return allowed or not is_premium_model(str(entry.get("id", "")))

    def _annotate(entry: dict) -> dict:
        annotated = dict(entry)
        annotated["tier"] = "premium" if is_premium_model(str(entry.get("id", ""))) else "free"
        return annotated

    if isinstance(catalog, dict):
        result = dict(catalog)
        result["models"] = [_annotate(m) for m in (catalog.get("models") or []) if _usable(m)]
        result["user_can_use_premium"] = allowed
        return result

    if isinstance(catalog, list):
        return [_annotate(m) for m in catalog if _usable(m)]

    return catalog
