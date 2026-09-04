"""Multi-user bearer token verification backed by the shared registry.

Each colleague authenticates with a personal ``cv_odoo_…`` key; the sha256
of the presented token is matched against the registry (CLORAG-managed
users.db). The optional static MCP_API_KEY keeps working as a fallback and
maps to a synthetic admin identity that uses the env-based Odoo client.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac

from fastmcp.server.auth import AccessToken, TokenVerifier

from .users_db import UsersDb

ENV_ADMIN_CLIENT_ID = "env-admin"


class DbTokenVerifier(TokenVerifier):
    """Verify bearer tokens against the shared per-user registry."""

    def __init__(self, users_db: UsersDb, static_api_key: str | None = None) -> None:
        super().__init__()
        self._db = users_db
        self._static = static_api_key

    async def verify_token(self, token: str) -> AccessToken | None:
        if self._static and hmac.compare_digest(token, self._static):
            return AccessToken(
                token=token,
                client_id=ENV_ADMIN_CLIENT_ID,
                scopes=["read", "write"],
                claims={"role": "admin", "auth": "static"},
            )
        key_hash = hashlib.sha256(token.encode()).hexdigest()
        # SQLite lookup off the event loop
        identity = await asyncio.to_thread(self._db.lookup_api_key, key_hash)
        if identity is None:
            return None
        scopes = ["read"] if identity.role == "readonly" else ["read", "write"]
        return AccessToken(
            token=token,
            client_id=identity.user_id,
            scopes=scopes,
            claims={
                "role": identity.role,
                "name": identity.name,
                "email": identity.email,
                "auth": "registry",
            },
        )
