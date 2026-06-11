"""Shared fixtures for unit tests (no live Odoo)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

TEST_ENCRYPTION_KEY = "test-encryption-key"

# Exact contract DDL — mirrors clorag core/user_db.py
_REGISTRY_DDL = """
CREATE TABLE users (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL,
    role TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE api_keys (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
    server TEXT NOT NULL, key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL, created_at TEXT NOT NULL,
    last_used_at TEXT, revoked_at TEXT);
CREATE TABLE user_odoo_credentials (
    user_id TEXT PRIMARY KEY REFERENCES users(id),
    odoo_username TEXT NOT NULL, encrypted_secret TEXT NOT NULL,
    updated_at TEXT NOT NULL);
CREATE TABLE user_skills (
    user_id TEXT NOT NULL REFERENCES users(id),
    skill_name TEXT NOT NULL, PRIMARY KEY (user_id, skill_name));
"""


def encrypt_with_contract(data: dict, salt: bytes, password: str) -> str:
    """Encrypt exactly like clorag's token_encryption.py (the contract)."""
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480_000)
    fernet = Fernet(base64.urlsafe_b64encode(kdf.derive(password.encode())))
    return fernet.encrypt(json.dumps(data).encode()).decode()


class RegistrySeed:
    """Handle on a seeded test registry."""

    def __init__(self, db_path: Path, salt: bytes):
        self.db_path = db_path
        self.salt = salt
        self.keys: dict[str, str] = {}      # label -> full key
        self.user_ids: dict[str, str] = {}  # label -> user id
        self.key_ids: dict[str, str] = {}   # label -> key id

    def add_user(self, conn, label, name, role, is_active=True):
        user_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, name, f"{label}@cyanview.com", role, int(is_active), now, now),
        )
        self.user_ids[label] = user_id
        return user_id

    def add_key(self, conn, label, user_id, server, revoked=False):
        full_key = f"cv_{server}_{uuid.uuid4().hex}"
        key_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO api_keys VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
            (key_id, user_id, server,
             hashlib.sha256(full_key.encode()).hexdigest(),
             full_key[:12], now, now if revoked else None),
        )
        self.keys[label] = full_key
        self.key_ids[label] = key_id
        return full_key


@pytest.fixture()
def users_db_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RegistrySeed:
    """A seeded registry with salt, users, keys, credentials and skills."""
    db_path = tmp_path / "users.db"
    salt = os.urandom(16)
    (tmp_path / ".token_salt").write_bytes(salt)
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", TEST_ENCRYPTION_KEY)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY_FILE", raising=False)

    seed = RegistrySeed(db_path, salt)
    conn = sqlite3.connect(db_path)
    conn.executescript(_REGISTRY_DDL)

    admin_id = seed.add_user(conn, "admin", "Alan Admin", "admin")
    member_id = seed.add_user(conn, "member", "Thierry Support", "support")
    ro_id = seed.add_user(conn, "readonly", "Read Only", "readonly")
    inactive_id = seed.add_user(conn, "inactive", "Gone User", "support", is_active=False)

    seed.add_key(conn, "admin_odoo", admin_id, "odoo")
    seed.add_key(conn, "member_odoo", member_id, "odoo")
    seed.add_key(conn, "readonly_odoo", ro_id, "odoo")
    seed.add_key(conn, "inactive_odoo", inactive_id, "odoo")
    seed.add_key(conn, "member_revoked", member_id, "odoo", revoked=True)
    seed.add_key(conn, "member_clorag", member_id, "clorag")  # must NOT match server='odoo'

    now = datetime.now().isoformat()
    conn.execute(
        "INSERT INTO user_odoo_credentials VALUES (?, ?, ?, ?)",
        (member_id, "thierry@cyanview.com",
         encrypt_with_contract({"api_key": "thierry-odoo-key"}, salt, TEST_ENCRYPTION_KEY),
         now),
    )
    conn.executemany(
        "INSERT INTO user_skills VALUES (?, ?)",
        [(member_id, "cyanview-rma"), (member_id, "cyanview-serial-tracker")],
    )
    conn.commit()
    conn.close()
    return seed
