"""One server answers both MCP eras: sessionless 2026-07-28 and session-based 2025-11-25.

FastMCP 4 negotiates the protocol per connection, which is the whole premise of
migrating without a flag day: a new client gets the new protocol, an old one keeps
working. These tests drive the real FastMCP app through the in-memory client in
both modes, with the Odoo client stubbed.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from fastmcp import Client

import odoo_mcp.app  # noqa: F401  (binds the mcp decorator first)
from odoo_mcp import server  # noqa: F401  (registers tools and resources)
from odoo_mcp.app import mcp

MODERN = "2026-07-28"
LEGACY = "2025-11-25"
TOOLS = {"execute_method", "batch_execute", "execute_workflow", "read_resource"}
FIELDS = {"name": {"type": "char", "string": "Name"}}


def _client(era: str) -> Client:
    return Client(mcp, mode="legacy") if era == LEGACY else Client(mcp)


@pytest.fixture
def odoo(monkeypatch):
    stub = MagicMock()
    stub.url = "https://odoo.example"
    stub.username = "tester"
    stub.execute_method.side_effect = lambda model, method, *args, **kwargs: (
        FIELDS if method == "fields_get" else [{"id": 1, "name": "x"}]
    )
    for module in ("odoo_mcp.server", "odoo_mcp.resources"):
        monkeypatch.setattr(f"{module}.get_odoo_client", lambda: stub)
    # The lifespan validates the env client at startup, before any caller exists.
    monkeypatch.setattr("odoo_mcp.app.get_env_client", lambda: stub)
    return stub


def _run(era: str, body):
    async def main():
        async with _client(era) as client:
            return await body(client)

    return asyncio.run(main())


@pytest.mark.parametrize("era", [MODERN, LEGACY])
def test_each_era_is_negotiated(odoo, era):
    assert _run(era, lambda client: asyncio.sleep(0, client.protocol_version)) == era


@pytest.mark.parametrize("era", [MODERN, LEGACY])
def test_surface_is_the_same_in_both_eras(odoo, era):
    async def body(client):
        tools = await client.list_tools()
        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        prompts = await client.list_prompts()
        return {t.name for t in tools}, len(resources) + len(templates), len(prompts)

    tools, resource_count, prompt_count = _run(era, body)

    assert tools == TOOLS
    assert resource_count == 38
    assert prompt_count == 12


@pytest.mark.parametrize("era", [MODERN, LEGACY])
@pytest.mark.parametrize(
    "uri",
    [
        "odoo://domain-syntax",
        "odoo://model/res.partner/quick-schema",
        "odoo://model/account.move.line/fields",
        "odoo://bundle/res.partner,sale.order",
        "odoo://methods/res.partner",
        "odoo://find-model/customer",
    ],
)
def test_templated_resources_accept_dotted_and_comma_separated_parameters(odoo, era, uri):
    """FastMCP 4 screens template parameters for path traversal; model names must pass."""
    contents = _run(era, lambda client: client.read_resource(uri))

    assert contents and contents[0].text


@pytest.mark.parametrize("era", [MODERN, LEGACY])
@pytest.mark.parametrize("uri", ["odoo://model/../quick-schema", "odoo://model/res.partner%2F..%2Fx/fields"])
def test_a_malformed_model_parameter_never_reaches_odoo(odoo, era, uri):
    """Whether FastMCP's screen or _validate_model stops it, no request may go out."""

    async def body(client):
        try:
            contents = await client.read_resource(uri)
        except Exception as exc:  # refused by the framework
            return f"refused: {exc}"
        return contents[0].text

    outcome = _run(era, body)

    assert "refused" in outcome or "error" in outcome.lower()
    odoo.execute_method.assert_not_called()


@pytest.mark.parametrize("era", [MODERN, LEGACY])
def test_sync_tool_runs(odoo, era):
    arguments = {"model": "res.partner", "method": "search_read", "kwargs_json": '{"domain": [], "limit": 1}'}

    result = _run(era, lambda client: client.call_tool("execute_method", arguments))

    assert result.data.success is True
    assert result.data.result == [{"id": 1, "name": "x"}]


@pytest.mark.parametrize("era", [MODERN, LEGACY])
def test_task_enabled_tool_runs(odoo, era):
    """batch_execute is task=True: it needs the tasks extension registered in app.py."""
    operations = [{"model": "res.partner", "method": "search_read", "kwargs_json": '{"domain": []}'}]

    result = _run(era, lambda client: client.call_tool("batch_execute", {"operations": operations}))

    assert result.data.success is True
    assert result.data.successful_operations == 1


def test_task_enabled_tool_runs_as_a_background_task(odoo):
    from fastmcp_tasks import call_tool_task

    operations = [{"model": "res.partner", "method": "search_read", "kwargs_json": '{"domain": []}'}]

    async def body(client):
        task = await call_tool_task(client, "batch_execute", {"operations": operations})
        return await task.result()

    result = _run(MODERN, body)

    assert result.data.success is True
