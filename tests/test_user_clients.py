"""Unit tests for per-user OdooClient resolution (user_clients.py)."""

import sqlite3
from datetime import datetime
from types import SimpleNamespace

import pytest

import odoo_mcp.user_clients as user_clients
from odoo_mcp.users_db import UsersDb
from tests.conftest import TEST_ENCRYPTION_KEY, encrypt_with_contract


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch, users_db_seed):
    """Fresh cache, registry env, and global Odoo config for each test."""
    monkeypatch.setattr(user_clients, "_cache", {})
    monkeypatch.setenv("USERS_DB_PATH", str(users_db_seed.db_path))
    monkeypatch.setenv("ODOO_URL", "https://odoo.example.com")
    monkeypatch.setenv("ODOO_DB", "cyanview")
    # reset users_db singleton so it picks the temp path
    import odoo_mcp.users_db as users_db_module

    monkeypatch.setattr(users_db_module, "_users_db", None)
    yield


def _fake_token(client_id, role="support", name="Test"):
    return SimpleNamespace(client_id=client_id, claims={"role": role, "name": name})


def test_no_token_returns_none(monkeypatch, users_db_seed):
    monkeypatch.setattr(user_clients, "_safe_get_access_token", lambda: None)
    assert user_clients.get_client_for_current_user() is None
    assert user_clients.current_role() is None


def test_env_admin_returns_none(monkeypatch, users_db_seed):
    monkeypatch.setattr(
        user_clients, "_safe_get_access_token",
        lambda: _fake_token("env-admin", role="admin"),
    )
    assert user_clients.get_client_for_current_user() is None
    assert user_clients.current_role() == "admin"


def test_registry_user_gets_personal_client(monkeypatch, users_db_seed):
    member_id = users_db_seed.user_ids["member"]
    monkeypatch.setattr(
        user_clients, "_safe_get_access_token", lambda: _fake_token(member_id)
    )
    client = user_clients.get_client_for_current_user()
    assert client is not None
    assert client.username == "thierry@cyanview.com"
    assert client.auth_credential == "thierry-odoo-key"
    assert client.url == "https://odoo.example.com"
    assert client.db == "cyanview"


def test_client_is_cached(monkeypatch, users_db_seed):
    member_id = users_db_seed.user_ids["member"]
    monkeypatch.setattr(
        user_clients, "_safe_get_access_token", lambda: _fake_token(member_id)
    )
    first = user_clients.get_client_for_current_user()
    second = user_clients.get_client_for_current_user()
    assert first is second


def test_credential_rotation_rebuilds_client(monkeypatch, users_db_seed):
    member_id = users_db_seed.user_ids["member"]
    monkeypatch.setattr(
        user_clients, "_safe_get_access_token", lambda: _fake_token(member_id)
    )
    first = user_clients.get_client_for_current_user()

    # Rotate credentials in the registry (as CLORAG would)
    conn = sqlite3.connect(users_db_seed.db_path)
    conn.execute(
        "UPDATE user_odoo_credentials SET encrypted_secret = ?, updated_at = ?"
        " WHERE user_id = ?",
        (
            encrypt_with_contract(
                {"api_key": "rotated-key"}, users_db_seed.salt, TEST_ENCRYPTION_KEY
            ),
            datetime.now().isoformat() + "-rotated",
            member_id,
        ),
    )
    conn.commit()
    conn.close()

    # Expire the TTL
    user_clients._cache[member_id].checked_at = 0.0
    second = user_clients.get_client_for_current_user()
    assert second is not first
    assert second.auth_credential == "rotated-key"


def test_missing_credentials_raises_permission_error(monkeypatch, users_db_seed):
    admin_id = users_db_seed.user_ids["admin"]  # admin has no stored credentials
    monkeypatch.setattr(
        user_clients, "_safe_get_access_token",
        lambda: _fake_token(admin_id, role="admin", name="Alan Admin"),
    )
    with pytest.raises(PermissionError, match="Alan Admin"):
        user_clients.get_client_for_current_user()


def test_dispatcher_falls_back_to_env_singleton(monkeypatch, users_db_seed):
    """get_odoo_client() uses the env singleton when no token is present."""
    import odoo_mcp.odoo_client as odoo_client_module

    monkeypatch.setattr(user_clients, "_safe_get_access_token", lambda: None)
    sentinel = object()
    monkeypatch.setattr(odoo_client_module, "_get_env_client", lambda: sentinel)
    assert odoo_client_module.get_odoo_client() is sentinel


def test_dispatcher_prefers_user_client(monkeypatch, users_db_seed):
    import odoo_mcp.odoo_client as odoo_client_module

    member_id = users_db_seed.user_ids["member"]
    monkeypatch.setattr(
        user_clients, "_safe_get_access_token", lambda: _fake_token(member_id)
    )
    client = odoo_client_module.get_odoo_client()
    assert client.username == "thierry@cyanview.com"
