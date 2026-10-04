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

from fastmcp.server.auth import AccessToken

from .auth_verifier import ENV_ADMIN_CLIENT_ID, STATIC_KEY_AUTH
from .odoo_client import OdooClient
from .token_crypto import decrypt_secret
from .users_db import UsersDb, get_users_db

_TTL_SECONDS = 300.0

_cache: dict[str, _Entry] = {}
_cache_lock = threading.Lock()

# Set once, by app._get_auth_provider, when it installs the registry verifier: the
# registry that authenticates callers is the one their Odoo credentials are read from,
# whatever the environment says later (load_config() reloads .env files at runtime).
_registry: UsersDb | None = None

# Set once, by the entry point, when it serves MCP over STDIO (``__main__.main``).
# Every other start (HTTP through __main__, ``fastmcp run``, an ASGI server on
# ``mcp.http_app()``) leaves it False: the fail-closed side.
_serving_stdio = False


@dataclass
class _Entry:
    client: OdooClient
    creds_updated_at: str
    checked_at: float


def _safe_get_access_token() -> AccessToken | None:
    """Current FastMCP access token, or None (stdio / no auth context)."""
    try:
        from fastmcp.server.dependencies import get_access_token

        return get_access_token()
    except Exception:
        return None


def declare_registry_auth(users_db: UsersDb) -> None:
    """Record the registry the installed token verifier authenticates against."""
    global _registry
    _registry = users_db


def _active_registry() -> UsersDb | None:
    """The registry installed at startup, else the one USERS_DB_PATH names now."""
    return _registry if _registry is not None else get_users_db()


def declare_stdio_transport() -> None:
    """Record that this process serves STDIO, where no caller authenticates.

    Called by the entry point right before ``mcp.run()``; never reset. Taken at
    startup rather than read from ``MCP_TRANSPORT`` so that no environment change
    (``load_config()`` reloads ``.env`` files) can reopen the check later.
    """
    global _serving_stdio
    _serving_stdio = True


def _registry_auth_enforced() -> bool:
    """True when callers authenticate against the registry: USERS_DB_PATH, not STDIO.

    There a missing access token means the caller's context was lost (a worker
    thread that did not inherit contextvars, a background task whose snapshot was
    not restored), never that the operator is calling.
    """
    return _active_registry() is not None and not _serving_stdio


def _is_static_key_identity(token: AccessToken) -> bool:
    """The static MCP_API_KEY identity: the only caller that runs on the env account.

    Recognised by its ``auth`` claim too, so a registry user whose id happens to
    read ``env-admin`` still gets their own client.
    """
    return bool(token.client_id == ENV_ADMIN_CLIENT_ID and token.claims.get("auth") == STATIC_KEY_AUTH)


def _caller_token() -> AccessToken | None:
    """Access token of the current caller; None only where no authentication exists.

    Raises:
        PermissionError: Registry mode without a token. A lost identity must fail
            closed instead of becoming role None (the privileged operator) on the
            env-configured Odoo account.
    """
    token = _safe_get_access_token()
    if token is None and _registry_auth_enforced():
        raise PermissionError(
            "No authenticated caller for this request: refusing to fall back to the server's own Odoo account."
        )
    return token


def current_role() -> str | None:
    """Role claim of the current caller, or None outside multi-user HTTP."""
    token = _caller_token()
    if token is None:
        return None
    return token.claims.get("role")


def get_client_for_current_user() -> OdooClient | None:
    """Personal client for the authenticated registry user.

    Returns None when the env singleton should be used instead (stdio mode,
    static-key fallback identity, or no registry configured).

    Raises:
        PermissionError: Authenticated registry user without stored Odoo
            credentials — actionable message for the caller — or no caller at
            all in registry mode (see ``_caller_token``).
    """
    token = _caller_token()
    if token is None or _is_static_key_identity(token):
        return None
    db = _active_registry()
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
            verify_ssl=os.environ.get("ODOO_VERIFY_SSL", "1").lower() in ("1", "true", "yes"),
        )
        _cache[user_id] = _Entry(client, creds.updated_at, now)
        return client
