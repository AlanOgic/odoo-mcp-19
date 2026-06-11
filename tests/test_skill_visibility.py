"""Unit tests for the per-user skill visibility middleware."""

import asyncio
from types import SimpleNamespace

import pytest
from fastmcp.exceptions import PromptError

import odoo_mcp.skill_visibility as sv
from odoo_mcp.skill_visibility import SkillVisibilityMiddleware
from odoo_mcp.users_db import UsersDb


def _prompt(name):
    return SimpleNamespace(name=name)


ALL_PROMPTS = [
    _prompt("odoo-exploration"),
    _prompt("search-records"),
    _prompt("cyanview-rma"),
    _prompt("cyanview-quote"),
    _prompt("cyanview-serial-tracker"),
]


def _middleware(users_db_seed):
    return SkillVisibilityMiddleware(UsersDb(users_db_seed.db_path))


def _list(mw, transport, token):
    """Run on_list_prompts with patched transport/token context."""

    async def run():
        sv_transport = sv.__dict__  # noqa: F841

        async def call_next(_ctx):
            return list(ALL_PROMPTS)

        ctx = SimpleNamespace(message=None)
        return await mw.on_list_prompts(ctx, call_next)

    return asyncio.run(run())


@pytest.fixture()
def patched_context(monkeypatch):
    """Patch the transport contextvar and access token lookups."""
    state = {"transport": "http", "token": None}

    class FakeVar:
        def get(self):
            return state["transport"]

    import fastmcp.server.context as fctx

    monkeypatch.setattr(fctx, "_current_transport", FakeVar())
    monkeypatch.setattr(sv, "get_access_token", lambda: state["token"])
    return state


def test_stdio_sees_everything(users_db_seed, patched_context):
    patched_context["transport"] = "stdio"
    result = _list(_middleware(users_db_seed), "stdio", None)
    assert len(result) == len(ALL_PROMPTS)


def test_no_token_fails_closed(users_db_seed, patched_context):
    patched_context["token"] = None
    result = _list(_middleware(users_db_seed), "http", None)
    names = {p.name for p in result}
    assert names == {"odoo-exploration", "search-records"}


def test_member_sees_only_allowed_skills(users_db_seed, patched_context):
    patched_context["token"] = SimpleNamespace(
        client_id=users_db_seed.user_ids["member"], claims={"role": "support"}
    )
    result = _list(_middleware(users_db_seed), "http", None)
    names = {p.name for p in result}
    assert names == {
        "odoo-exploration",
        "search-records",
        "cyanview-rma",
        "cyanview-serial-tracker",
    }


def test_admin_sees_everything(users_db_seed, patched_context):
    patched_context["token"] = SimpleNamespace(
        client_id=users_db_seed.user_ids["admin"], claims={"role": "admin"}
    )
    result = _list(_middleware(users_db_seed), "http", None)
    assert len(result) == len(ALL_PROMPTS)


def test_env_admin_sees_everything(users_db_seed, patched_context):
    patched_context["token"] = SimpleNamespace(
        client_id="env-admin", claims={"role": "admin"}
    )
    result = _list(_middleware(users_db_seed), "http", None)
    assert len(result) == len(ALL_PROMPTS)


def test_render_forbidden_skill_raises(users_db_seed, patched_context):
    patched_context["token"] = SimpleNamespace(
        client_id=users_db_seed.user_ids["member"], claims={"role": "support"}
    )
    mw = _middleware(users_db_seed)

    async def run():
        async def call_next(_ctx):
            return "rendered"

        ctx = SimpleNamespace(message=SimpleNamespace(name="cyanview-quote"))
        return await mw.on_get_prompt(ctx, call_next)

    with pytest.raises(PromptError, match="cyanview-quote"):
        asyncio.run(run())


def test_render_allowed_skill_passes(users_db_seed, patched_context):
    patched_context["token"] = SimpleNamespace(
        client_id=users_db_seed.user_ids["member"], claims={"role": "support"}
    )
    mw = _middleware(users_db_seed)

    async def run():
        async def call_next(_ctx):
            return "rendered"

        ctx = SimpleNamespace(message=SimpleNamespace(name="cyanview-rma"))
        return await mw.on_get_prompt(ctx, call_next)

    assert asyncio.run(run()) == "rendered"


def test_render_generic_prompt_never_blocked(users_db_seed, patched_context):
    patched_context["token"] = None
    mw = _middleware(users_db_seed)

    async def run():
        async def call_next(_ctx):
            return "rendered"

        ctx = SimpleNamespace(message=SimpleNamespace(name="odoo-exploration"))
        return await mw.on_get_prompt(ctx, call_next)

    assert asyncio.run(run()) == "rendered"
