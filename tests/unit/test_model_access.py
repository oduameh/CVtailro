"""Unit tests — premium model access helpers (no DB, no network)."""

from __future__ import annotations

import pytest

from app.models.user import User
from app.services import model_access
from app.services.admin_config import AdminConfigManager
from config import RECOMMENDED_MODELS

pytestmark = pytest.mark.unit

FREE_ID = "nvidia/nemotron-3-super-120b-a12b:free"
PAID_ID = "openai/gpt-4o-mini"
AUTO_FREE_ID = "openrouter/free"  # free group entry whose ID lacks the ':free' suffix


def _enable_switch(monkeypatch, enabled=True):
    monkeypatch.setattr(AdminConfigManager, "get", staticmethod(lambda key: "true" if enabled else "false"))


def _make_user(is_admin=False, granted=False) -> User:
    return User(
        email="premium-test@example.com",
        name="Premium Test",
        is_admin=is_admin,
        can_use_premium_models=granted,
    )


class TestIsPremiumModel:
    def test_free_suffix_model_is_not_premium(self):
        assert model_access.is_premium_model(FREE_ID) is False

    def test_paid_model_is_premium(self):
        assert model_access.is_premium_model(PAID_ID) is True

    def test_auto_router_free_entry_is_not_premium(self):
        assert model_access.is_premium_model(AUTO_FREE_ID) is False

    def test_unknown_model_is_not_premium(self):
        assert model_access.is_premium_model("totally/made-up") is False

    def test_every_paid_group_entry_is_flagged(self):
        paid_names = [name for name in RECOMMENDED_MODELS if "(Free)" not in name]
        assert paid_names
        for name in paid_names:
            assert model_access.is_premium_model(RECOMMENDED_MODELS[name]) is True

    def test_every_free_group_entry_is_not_flagged(self):
        free_names = [name for name in RECOMMENDED_MODELS if "(Free)" in name]
        assert free_names
        for name in free_names:
            assert model_access.is_premium_model(RECOMMENDED_MODELS[name]) is False


class TestCanUsePremiumModels:
    def test_anonymous_denied_when_enabled(self, monkeypatch):
        _enable_switch(monkeypatch)
        assert model_access.can_use_premium_models(None) is False

    def test_free_user_denied(self, monkeypatch):
        _enable_switch(monkeypatch)
        assert model_access.can_use_premium_models(_make_user()) is False

    def test_granted_user_allowed(self, monkeypatch):
        _enable_switch(monkeypatch)
        assert model_access.can_use_premium_models(_make_user(granted=True)) is True

    def test_admin_allowed_without_grant(self, monkeypatch):
        _enable_switch(monkeypatch)
        assert model_access.can_use_premium_models(_make_user(is_admin=True)) is True

    def test_kill_switch_off_denies_granted_user(self, monkeypatch):
        _enable_switch(monkeypatch, enabled=False)
        assert model_access.can_use_premium_models(_make_user(granted=True)) is False

    def test_kill_switch_off_denies_admin(self, monkeypatch):
        _enable_switch(monkeypatch, enabled=False)
        assert model_access.can_use_premium_models(_make_user(is_admin=True)) is False

    def test_unset_setting_defaults_to_enabled(self, monkeypatch):
        monkeypatch.setattr(AdminConfigManager, "get", staticmethod(lambda key: None))
        assert model_access.can_use_premium_models(_make_user(granted=True)) is True

    def test_setting_explicitly_false_disables(self, monkeypatch):
        monkeypatch.setattr(AdminConfigManager, "get", staticmethod(lambda key: "false"))
        assert model_access.can_use_premium_models(_make_user(granted=True)) is False


class TestFilterCatalogForUser:
    @staticmethod
    def _catalog():
        return {
            "models": [
                {"id": FREE_ID, "name": "Nemotron 3 Super 120B (Free)"},
                {"id": PAID_ID, "name": "GPT-4o Mini"},
            ],
            "default": FREE_ID,
            "user_selectable": True,
        }

    def test_anonymous_gets_free_only_with_tiers_and_flag(self, monkeypatch):
        _enable_switch(monkeypatch)
        out = model_access.filter_catalog_for_user(self._catalog(), None)

        assert isinstance(out, dict)
        assert out["user_can_use_premium"] is False
        assert [m["id"] for m in out["models"]] == [FREE_ID]
        assert all(m["tier"] == "free" for m in out["models"])
        assert out["default"] == FREE_ID and out["user_selectable"] is True

    def test_granted_user_sees_premium_with_tier_annotations(self, monkeypatch):
        _enable_switch(monkeypatch)
        out = model_access.filter_catalog_for_user(self._catalog(), _make_user(granted=True))

        assert out["user_can_use_premium"] is True
        tiers = {m["id"]: m["tier"] for m in out["models"]}
        assert tiers == {FREE_ID: "free", PAID_ID: "premium"}

    def test_kill_switch_hides_premium_even_from_granted_user(self, monkeypatch):
        _enable_switch(monkeypatch, enabled=False)
        out = model_access.filter_catalog_for_user(self._catalog(), _make_user(granted=True))

        assert out["user_can_use_premium"] is False
        assert [m["id"] for m in out["models"]] == [FREE_ID]

    def test_list_input_returns_annotated_list(self, monkeypatch):
        _enable_switch(monkeypatch)
        models = [{"id": FREE_ID, "name": "Nemotron (Free)"}, {"id": PAID_ID, "name": "GPT-4o Mini"}]

        denied = model_access.filter_catalog_for_user(models, None)
        assert isinstance(denied, list)
        assert [m["id"] for m in denied] == [FREE_ID]

        allowed = model_access.filter_catalog_for_user(models, _make_user(is_admin=True))
        assert [m["tier"] for m in allowed] == ["free", "premium"]

    def test_source_entries_are_not_mutated(self, monkeypatch):
        _enable_switch(monkeypatch)
        catalog = self._catalog()
        model_access.filter_catalog_for_user(catalog, None)

        assert all("tier" not in m and "user_can_use_premium" not in m for m in catalog["models"])
        assert "user_can_use_premium" not in catalog
