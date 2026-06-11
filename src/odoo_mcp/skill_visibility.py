"""Per-user visibility of the Cyanview skill prompts.

Filters ``cyanview-*`` prompts in list_prompts according to the user's
allowlist in the shared registry (user_skills), and enforces the same rule
at render time. Generic Odoo prompts stay visible to everyone. Stdio mode
(Alan local) and admin/static identities see everything.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import mcp.types as mt
from fastmcp.exceptions import PromptError
from fastmcp.prompts.base import Prompt
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

from .auth_verifier import ENV_ADMIN_CLIENT_ID
from .users_db import UsersDb

CYANVIEW_PREFIX = "cyanview-"


class SkillVisibilityMiddleware(Middleware):
    """Filter cyanview-* prompts by the authenticated user's skill allowlist."""

    def __init__(self, users_db: UsersDb) -> None:
        self._db = users_db

    def _allowed_skills(self) -> frozenset[str] | None:
        """None = unrestricted (stdio, admin role, static env-admin key)."""
        from fastmcp.server.context import _current_transport

        if _current_transport.get() == "stdio":
            return None  # Alan local: everything visible, unchanged
        token = get_access_token()
        if token is None:
            return frozenset()  # fail closed
        if token.client_id == ENV_ADMIN_CLIENT_ID or token.claims.get("role") == "admin":
            return None
        return self._db.get_skills(token.client_id)

    async def on_list_prompts(
        self,
        context: MiddlewareContext[mt.ListPromptsRequest],
        call_next: CallNext[mt.ListPromptsRequest, Sequence[Prompt]],
    ) -> Sequence[Prompt]:
        prompts = await call_next(context)
        allowed = await asyncio.to_thread(self._allowed_skills)
        if allowed is None:
            return prompts
        return [
            p
            for p in prompts
            if not p.name.startswith(CYANVIEW_PREFIX) or p.name in allowed
        ]

    async def on_get_prompt(
        self,
        context: MiddlewareContext[mt.GetPromptRequestParams],
        call_next,  # type: ignore[no-untyped-def]
    ):
        name = context.message.name
        if name.startswith(CYANVIEW_PREFIX):
            allowed = await asyncio.to_thread(self._allowed_skills)
            if allowed is not None and name not in allowed:
                raise PromptError(
                    f"The '{name}' skill is not enabled for your profile."
                    " Ask an administrator to grant it in the CLORAG user"
                    " registry (/admin/users)."
                )
        return await call_next(context)
