"""Per-user OdooClient resolution — "chacun son Odoo".

Maps the authenticated registry user (FastMCP access token) to an
OdooClient built from THEIR Odoo credentials, so writes are attributed to
the real Odoo account. Clients are cached per user with a TTL re-check of
the registry's ``updated_at`` so credential rotation propagates without a
restart while the requests.Session is reused when nothing changed.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from .auth_verifier import ENV_ADMIN_CLIENT_ID
from .odoo_client import OdooClient
from .token_crypto import decrypt_secret
from .users_db import get_users_db

_TTL_SECONDS = 300.0

_cache: dict[str, _Entry] = {}
_cache_lock = threading.Lock()


@dataclass
class _Entry:
    client: OdooClient
    creds_updated_at: str
    checked_at: float


def _safe_get_access_token():
    """Current FastMCP access token, or None (stdio / no auth context)."""
    try:
        from fastmcp.server.dependencies import get_access_token

        return get_access_token()
    except Exception:
        return None


def current_role() -> str | None:
    """Role claim of the current caller, or None outside multi-user HTTP."""
    token = _safe_get_access_token()
    if token is None:
        return None
    return token.claims.get("role")


def get_client_for_current_user() -> OdooClient | None:
    """Personal client for the authenticated registry user.

    Returns None when the env singleton should be used instead (stdio mode,
    static-key fallback identity, or no registry configured).

    Raises:
        PermissionError: Authenticated registry user without stored Odoo
            credentials — actionable message for the caller.
    """
    token = _safe_get_access_token()
    if token is None or token.client_id == ENV_ADMIN_CLIENT_ID:
        return None
    db = get_users_db()
    if db is None:
        return None

    user_id = token.client_id
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(user_id)
        if entry is not None and now - entry.checked_at < _TTL_SECONDS:
            return entry.client

    creds = db.get_odoo_credentials(user_id)
    if creds is None:
        raise PermissionError(
            f"No Odoo credentials registered for user '{token.claims.get('name', user_id)}'."
            " Ask an admin to add them in the CLORAG user registry (/admin/users)."
        )

    with _cache_lock:
        entry = _cache.get(user_id)
        if entry is not None and entry.creds_updated_at == creds.updated_at:
            # Credentials unchanged: refresh the TTL, keep the Session alive.
            entry.checked_at = now
            return entry.client

        secret = decrypt_secret(creds.encrypted_secret, db_path=db.path)
        client = OdooClient(
            url=os.environ["ODOO_URL"],
            db=os.environ["ODOO_DB"],
            username=creds.odoo_username,
            api_key=str(secret["api_key"]),
            timeout=int(os.environ.get("ODOO_TIMEOUT", "30")),
            verify_ssl=os.environ.get("ODOO_VERIFY_SSL", "1").lower()
            in ("1", "true", "yes"),
        )
        _cache[user_id] = _Entry(client, creds.updated_at, now)
        return client
