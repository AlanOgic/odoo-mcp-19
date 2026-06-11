"""Unit tests for DbTokenVerifier and UsersDb lookups."""

import asyncio
import hashlib

from odoo_mcp.auth_verifier import ENV_ADMIN_CLIENT_ID, DbTokenVerifier
from odoo_mcp.users_db import UsersDb


def _verify(verifier, token):
    return asyncio.run(verifier.verify_token(token))


def test_valid_key_returns_identity(users_db_seed):
    db = UsersDb(users_db_seed.db_path)
    verifier = DbTokenVerifier(db)
    token = _verify(verifier, users_db_seed.keys["member_odoo"])
    assert token is not None
    assert token.client_id == users_db_seed.user_ids["member"]
    assert token.claims["role"] == "support"
    assert token.claims["auth"] == "registry"
    assert "write" in token.scopes


def test_readonly_role_gets_read_scope_only(users_db_seed):
    db = UsersDb(users_db_seed.db_path)
    verifier = DbTokenVerifier(db)
    token = _verify(verifier, users_db_seed.keys["readonly_odoo"])
    assert token is not None
    assert token.scopes == ["read"]


def test_revoked_key_rejected(users_db_seed):
    verifier = DbTokenVerifier(UsersDb(users_db_seed.db_path))
    assert _verify(verifier, users_db_seed.keys["member_revoked"]) is None


def test_inactive_user_rejected(users_db_seed):
    verifier = DbTokenVerifier(UsersDb(users_db_seed.db_path))
    assert _verify(verifier, users_db_seed.keys["inactive_odoo"]) is None


def test_clorag_key_rejected_on_odoo_server(users_db_seed):
    verifier = DbTokenVerifier(UsersDb(users_db_seed.db_path))
    assert _verify(verifier, users_db_seed.keys["member_clorag"]) is None


def test_unknown_token_rejected(users_db_seed):
    verifier = DbTokenVerifier(UsersDb(users_db_seed.db_path))
    assert _verify(verifier, "cv_odoo_deadbeef") is None


def test_static_fallback_maps_to_env_admin(users_db_seed):
    verifier = DbTokenVerifier(
        UsersDb(users_db_seed.db_path), static_api_key="shared-static"
    )
    token = _verify(verifier, "shared-static")
    assert token is not None
    assert token.client_id == ENV_ADMIN_CLIENT_ID
    assert token.claims["auth"] == "static"


def test_users_db_is_read_only(users_db_seed):
    """The mode=ro URI must prevent any write from this process."""
    import sqlite3

    import pytest

    db = UsersDb(users_db_seed.db_path)
    with pytest.raises(sqlite3.OperationalError):
        with db._connect() as conn:
            conn.execute("DELETE FROM users")


def test_lookup_uses_sha256(users_db_seed):
    """Sanity: the registry stores hashes, not keys."""
    import sqlite3

    full_key = users_db_seed.keys["member_odoo"]
    conn = sqlite3.connect(users_db_seed.db_path)
    rows = conn.execute("SELECT key_hash FROM api_keys").fetchall()
    conn.close()
    hashes = {row[0] for row in rows}
    assert full_key not in hashes
    assert hashlib.sha256(full_key.encode()).hexdigest() in hashes


def test_get_skills(users_db_seed):
    db = UsersDb(users_db_seed.db_path)
    skills = db.get_skills(users_db_seed.user_ids["member"])
    assert skills == frozenset({"cyanview-rma", "cyanview-serial-tracker"})
    assert db.get_skills(users_db_seed.user_ids["admin"]) == frozenset()


def test_get_odoo_credentials(users_db_seed):
    db = UsersDb(users_db_seed.db_path)
    creds = db.get_odoo_credentials(users_db_seed.user_ids["member"])
    assert creds is not None
    assert creds.odoo_username == "thierry@cyanview.com"
    assert db.get_odoo_credentials(users_db_seed.user_ids["admin"]) is None
