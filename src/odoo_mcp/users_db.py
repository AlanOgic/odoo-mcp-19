"""Read-only access to the shared per-user registry (users.db).

The registry is owned and written by CLORAG (its admin page at
/admin/users); this server only reads it through a shared Docker volume.
Connections are opened with ``mode=ro`` so this process can never write,
even if the mount is read-write. Schema contract — never change it here:

    users(id, name, email, role, is_active, created_at, updated_at)
    api_keys(id, user_id, server, key_hash, key_prefix, created_at,
             last_used_at, revoked_at)
    user_odoo_credentials(user_id, odoo_username, encrypted_secret, updated_at)
    user_skills(user_id, skill_name)
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiKeyIdentity:
    """The user owning a verified API key."""

    user_id: str
    name: str
    email: str
    role: str


@dataclass(frozen=True)
class OdooCredentials:
    """A user's stored Odoo credentials (secret still encrypted)."""

    user_id: str
    odoo_username: str
    encrypted_secret: str
    updated_at: str


class UsersDb:
    """Short-lived read-only connections to the registry."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def lookup_api_key(self, key_hash: str) -> ApiKeyIdentity | None:
        """Find the active user owning a non-revoked 'odoo' key by hash."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT u.id, u.name, u.email, u.role"
                " FROM api_keys k JOIN users u ON u.id = k.user_id"
                " WHERE k.server = 'odoo' AND k.key_hash = ?"
                " AND k.revoked_at IS NULL AND u.is_active = 1",
                (key_hash,),
            ).fetchone()
        if row is None:
            return None
        return ApiKeyIdentity(
            user_id=row["id"], name=row["name"], email=row["email"], role=row["role"]
        )

    def get_odoo_credentials(self, user_id: str) -> OdooCredentials | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, odoo_username, encrypted_secret, updated_at"
                " FROM user_odoo_credentials WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return OdooCredentials(
            user_id=row["user_id"],
            odoo_username=row["odoo_username"],
            encrypted_secret=row["encrypted_secret"],
            updated_at=row["updated_at"],
        )

    def get_skills(self, user_id: str) -> frozenset[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT skill_name FROM user_skills WHERE user_id = ?", (user_id,)
            ).fetchall()
        return frozenset(row["skill_name"] for row in rows)


_users_db: UsersDb | None = None
_users_db_lock = threading.Lock()


def get_users_db() -> UsersDb | None:
    """Singleton registry handle from USERS_DB_PATH; None when unset."""
    global _users_db
    path = os.environ.get("USERS_DB_PATH")
    if not path:
        return None
    if _users_db is None or _users_db.path != Path(path):
        with _users_db_lock:
            if _users_db is None or _users_db.path != Path(path):
                _users_db = UsersDb(Path(path))
    return _users_db
