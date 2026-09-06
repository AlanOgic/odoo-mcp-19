"""Pin that blocking Odoo round-trips never run on the event-loop thread.

FastMCP 3.x dispatches handlers three different ways:

* sync ``@mcp.tool`` and *static* ``@mcp.resource`` bodies run in the anyio
  worker threadpool (non-blocking);
* *template* resources (``odoo://model/{model_name}/...``) whose body is a
  plain sync function are called **inline on the loop** by
  ``FunctionResourceTemplate.read`` — one slow ``fields_get`` freezes every
  session, ping included;
* ``async def`` tools that call the synchronous ``requests``-based
  ``OdooClient`` block the loop for the whole call.

The fixes register every parameterized resource as an ``async`` wrapper that
offloads the sync body, and route the Odoo calls inside ``batch_execute`` /
``execute_workflow`` through ``anyio.to_thread``. These tests fail loudly if a
future resource or tool regresses to loop-blocking dispatch.

No live Odoo: ``get_odoo_client`` is patched with a stub that records the
thread it was called from.
"""

import asyncio
import inspect
import json
import re
import threading
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client

import odoo_mcp.app as app
import odoo_mcp.resources as resources
import odoo_mcp.server as server
from odoo_mcp.app import mcp

_FIELDS = {
    "id": {"type": "integer", "readonly": True},
    "name": {"type": "char", "required": True},
}


def _recording_client(record: dict, **method_returns):
    """Stub OdooClient whose calls note the thread they ran on."""
    client = MagicMock()

    def _make(name, value):
        def _call(*args, **kwargs):
            record.setdefault(name, []).append(threading.get_ident())
            return value

        return _call

    for name, value in method_returns.items():
        getattr(client, name).side_effect = _make(name, value)
    return client


@contextmanager
def _stubbed(stub):
    """Serve ``stub`` to the resource handlers *and* to the FastMCP lifespan.

    ``Client(mcp)`` runs ``app_lifespan``, which builds the env client at
    startup; patching only ``resources`` would make these tests depend on a
    real ODOO_* config (CI deliberately has none).
    """
    with (
        patch.object(resources, "get_odoo_client", return_value=stub),
        patch.object(app, "get_odoo_client", return_value=stub),
    ):
        yield


def _token_from(text: str) -> str:
    match = re.search(r"confirmation_token='([^']+)'", text or "")
    assert match, f"no confirmation token in: {text!r}"
    return match.group(1)


# ----- template resources -----


def test_every_resource_template_is_registered_as_coroutine():
    """Only a coroutine template escapes FunctionResourceTemplate's inline call."""
    templates = asyncio.run(mcp.list_resource_templates(run_middleware=False))
    assert templates, "expected registered resource templates"
    blocking = [t.uri_template for t in templates if not inspect.iscoroutinefunction(t.fn)]
    assert blocking == [], f"sync template bodies run on the event loop: {blocking}"


def test_template_resource_read_runs_odoo_call_off_loop_thread():
    record: dict = {}
    stub = _recording_client(record, get_model_fields=_FIELDS)

    async def _read():
        async with Client(mcp) as client:
            contents = await client.read_resource("odoo://model/res.partner/quick-schema")
        return threading.get_ident(), contents

    with _stubbed(stub):
        loop_thread, contents = asyncio.run(_read())

    assert record["get_model_fields"], "fields_get was never called"
    assert all(tid != loop_thread for tid in record["get_model_fields"])
    payload = json.loads(contents[0].text)
    assert payload["model"] == "res.partner"
    assert payload["fields"]["name"]["req"] is True


def test_template_wrapper_returns_same_payload_as_sync_body():
    """The async registration must be a pure offload — no behaviour drift."""
    stub = _recording_client({}, get_model_fields=_FIELDS)

    async def _read():
        async with Client(mcp) as client:
            contents = await client.read_resource("odoo://model/res.partner/fields")
        return contents[0].text

    with _stubbed(stub):
        via_mcp = asyncio.run(_read())
        direct = resources.get_model_fields_light("res.partner")

    assert json.loads(via_mcp) == json.loads(direct)


# ----- async tools -----


def test_batch_execute_runs_odoo_calls_off_loop_thread(monkeypatch):
    monkeypatch.delenv("MCP_READ_ONLY", raising=False)
    record: dict = {}
    stub = _recording_client(record, execute_method=[{"id": 1, "name": "A"}])

    async def _run():
        response = await server.batch_execute(
            operations=[
                {"model": "res.partner", "method": "search_read", "kwargs_json": '{"fields": ["name"]}'},
                {"model": "res.partner", "method": "search_count", "kwargs_json": "{}"},
            ],
            progress=AsyncMock(),
        )
        return threading.get_ident(), response

    with patch.object(server, "get_odoo_client", return_value=stub):
        loop_thread, response = asyncio.run(_run())

    assert response.success is True, response.error
    assert len(record["execute_method"]) == 2
    assert all(tid != loop_thread for tid in record["execute_method"])


def test_execute_workflow_runs_odoo_calls_off_loop_thread(monkeypatch):
    monkeypatch.delenv("MCP_READ_ONLY", raising=False)
    monkeypatch.setenv("MCP_SAFETY_MODE", "strict")
    record: dict = {}
    stub = _recording_client(
        record,
        search_read=[{"id": 7, "type": "opportunity"}],
        execute_method=True,
    )

    async def _run():
        first = await server.execute_workflow(
            workflow="lead_to_won", params_json='{"lead_id": 7}', progress=AsyncMock()
        )
        if first.pending_confirmation:
            first = await server.execute_workflow(
                workflow="lead_to_won",
                params_json='{"lead_id": 7}',
                confirmed=True,
                confirmation_token=_token_from(first.tip),
                progress=AsyncMock(),
            )
        return threading.get_ident(), first

    with patch.object(server, "get_odoo_client", return_value=stub):
        loop_thread, response = asyncio.run(_run())

    assert response.success is True, response.error
    called = record.get("search_read", []) + record.get("execute_method", [])
    assert called, "workflow made no Odoo call"
    assert all(tid != loop_thread for tid in called)


@pytest.mark.parametrize("tool", [server.batch_execute, server.execute_workflow])
def test_async_tools_stay_async(tool):
    """Guard against 'fixing' the blocking by turning the task tools sync."""
    assert inspect.iscoroutinefunction(tool)
