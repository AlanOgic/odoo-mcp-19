"""Tool behaviours found wrong by driving the three tools against Odoo Online 19.3 (2026-11-26).

* the handshake reported FastMCP's version as the server's;
* both write-capable tools declared themselves non-destructive although they run ``unlink``;
* an empty batch surfaced Docket's "Total must be at least 1";
* a batched ``search_read`` without a limit returned the whole table (6 989 partners, 300 KB);
* "Field name error detected" was attached to version refusals and unknown models, while a
  real invalid field went through a useless search + read retry and got no field hint;
* a blocked call said ``pending_confirmation`` although no token will ever exist.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client

import odoo_mcp
import odoo_mcp.server as server
from odoo_mcp.app import mcp
from odoo_mcp.constants import DEFAULT_LIMIT, MAX_LIMIT

INVALID_FIELD_500 = (
    "Request failed: 500 INTERNAL SERVER ERROR for url: https://odoo.example/json/2/res.partner/search_read\n"
    "Odoo error: Invalid field res.partner.nope in condition ('nope', '=', 1)"
)
UNKNOWN_MODEL_404 = (
    "Request failed: 404 NOT FOUND for url: https://odoo.example/json/2/no.such_model/search\n"
    "Odoo error: the model 'no.such_model' does not exist"
)


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    for name in ("MCP_READ_ONLY", "MCP_WRITE_ALLOWLIST", "MCP_VALIDATE_PAYLOADS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MCP_SAFETY_MODE", "strict")
    monkeypatch.setattr(server, "current_role", lambda: None)


def _client(effect=None, version=(19, 3)):
    client = MagicMock()
    client.server_version.return_value = version
    if effect is not None:
        client.execute_method.side_effect = effect
    return client


def _call(client, **kwargs):
    with patch.object(server, "get_odoo_client", return_value=client):
        return server.execute_method(ctx=MagicMock(), **kwargs)


def _batch(client, operations, **kwargs):
    with patch.object(server, "get_odoo_client", return_value=client):
        return asyncio.run(server.batch_execute(operations=operations, progress=AsyncMock(), **kwargs))


def _raises(message):
    def _execute(*args, **kwargs):
        raise ValueError(message)

    return _execute


# ----- what the server says about itself -----


def _handshake(era="legacy"):
    stub = MagicMock()

    async def main():
        with patch("odoo_mcp.app.get_env_client", lambda: stub):
            async with Client(mcp, mode="legacy") if era == "legacy" else Client(mcp) as client:
                return client.server_info, await client.list_tools()

    return asyncio.run(main())


@pytest.mark.parametrize("era", ["legacy", "modern"])
def test_the_handshake_reports_this_servers_version_not_fastmcps(era):
    server_info, _tools = _handshake(era)
    assert server_info.version == odoo_mcp.__version__


def test_the_modern_era_lists_the_write_tools_as_destructive_too():
    _info, tools = _handshake("modern")
    assert {t.name: t.annotations.destructive_hint for t in tools if t.name != "read_resource"} == {
        "execute_method": True,
        "batch_execute": True,
    }


@pytest.mark.parametrize("tool", ["execute_method", "batch_execute"])
def test_write_capable_tools_are_declared_destructive(tool):
    _result, tools = _handshake()
    annotations = next(t for t in tools if t.name == tool).annotations
    assert annotations.destructive_hint is True
    assert annotations.read_only_hint is False


@pytest.mark.parametrize("tool", ["execute_method", "batch_execute"])
def test_confirmation_token_is_described(tool):
    _result, tools = _handshake()
    schema = next(t for t in tools if t.name == tool).input_schema
    assert "Single-use token" in schema["properties"]["confirmation_token"].get("description", "")


# ----- empty batch -----


def test_an_empty_batch_is_refused_with_a_message_naming_the_parameter():
    progress = AsyncMock()
    progress.set_total.side_effect = ValueError("Total must be at least 1")  # what Docket does
    client = _client()
    with patch.object(server, "get_odoo_client", return_value=client):
        response = asyncio.run(server.batch_execute(operations=[], progress=progress))

    assert response.success is False
    assert "'operations' is empty" in response.error
    assert response.total_operations == 0
    client.execute_method.assert_not_called()


# ----- a batched search has the limit of a single one -----


def _sent_kwargs(client):
    return [dict(call.kwargs) for call in client.execute_method.call_args_list]


def test_a_batched_search_read_without_a_limit_gets_the_default():
    client = _client(lambda *a, **k: [{"id": 1}])
    _batch(client, [{"model": "res.partner", "method": "search_read", "kwargs_json": '{"domain": []}'}])
    assert _sent_kwargs(client) == [{"domain": [], "limit": DEFAULT_LIMIT}]


def test_a_batched_search_limit_is_capped():
    client = _client(lambda *a, **k: [1])
    _batch(
        client, [{"model": "res.partner", "method": "search", "args_json": "[[]]", "kwargs_json": '{"limit": 1000000}'}]
    )
    assert _sent_kwargs(client) == [{"domain": [], "limit": MAX_LIMIT}]


def test_a_batched_positional_limit_and_double_wrapped_domain_are_normalised():
    client = _client(lambda *a, **k: [1])
    domain = '[["is_company", "=", true]]'
    _batch(client, [{"model": "res.partner", "method": "search", "args_json": f"[[{domain}], 0, 5000]"}])
    assert _sent_kwargs(client) == [{"domain": [["is_company", "=", True]], "offset": 0, "limit": MAX_LIMIT}]


def test_a_batched_search_count_gets_no_limit():
    client = _client(lambda *a, **k: 3)
    _batch(client, [{"model": "res.partner", "method": "search_count", "kwargs_json": '{"domain": []}'}])
    assert _sent_kwargs(client) == [{"domain": []}]


def test_a_batched_domain_given_twice_fails_that_operation():
    client = _client(lambda *a, **k: [1])
    op = {"model": "res.partner", "method": "search", "args_json": "[[]]", "kwargs_json": '{"domain": []}'}
    response = _batch(client, [op])
    assert response.results[0].success is False
    assert "Pass it once" in response.results[0].error
    client.execute_method.assert_not_called()


def test_a_batched_full_page_says_more_records_may_exist():
    client = _client(lambda *a, **k: [{"id": i} for i in range(2)])
    response = _batch(client, [{"model": "res.partner", "method": "search_read", "kwargs_json": '{"limit": 2}'}])
    assert response.results[0].success is True
    assert '"offset": 2' in response.results[0].note


def test_a_batched_short_page_and_a_non_search_carry_no_note():
    client = _client(lambda *a, **k: [{"id": 1}])
    response = _batch(
        client,
        [
            {"model": "res.partner", "method": "search_read", "kwargs_json": '{"limit": 5}'},
            {"model": "res.partner", "method": "read", "args_json": "[[1]]"},
        ],
    )
    assert [r.note for r in response.results] == [None, None]


def test_a_batched_search_with_a_non_integer_limit_fails_that_operation_only():
    client = _client(lambda *a, **k: 3)
    response = _batch(
        client,
        [
            {"model": "res.partner", "method": "search", "kwargs_json": '{"limit": "10"}'},
            {"model": "res.partner", "method": "search_count", "args_json": "[[]]"},
        ],
        atomic=False,
    )
    assert [r.success for r in response.results] == [False, True]
    assert "'limit' must be an integer" in response.results[0].error


# ----- hints say what actually went wrong -----


@pytest.mark.parametrize("method", ["read_group", "toggle_active"])
def test_a_version_refusal_is_not_called_a_field_error(method):
    response = _call(_client(), model="res.partner", method=method, args_json="[[1]]")
    assert response.success is False
    assert "does not exist on Odoo 19.3" in response.error
    assert "odoo://api/version-drift" in response.error  # the error carries the pointer itself
    assert response.hint is None


def test_an_unknown_model_points_at_model_discovery():
    response = _call(_client(_raises(UNKNOWN_MODEL_404)), model="no.such_model", method="search", args_json="[[]]")
    assert "Field name error" not in response.hint
    assert "odoo://find-model/" in response.hint


def test_a_missing_method_keeps_its_own_hint():
    error = "Request failed: 404 NOT FOUND\nOdoo error: The method 'ir.model.frobnicate' does not exist"
    response = _call(_client(_raises(error)), model="ir.model", method="search_count", args_json="[[]]")
    assert "odoo://methods/ir.model" in response.hint
    assert "find-model" not in response.hint


def test_a_missing_record_is_not_called_a_field_error():
    error = "Request failed: 404 NOT FOUND\nOdoo error: Record does not exist or has been deleted."
    response = _call(_client(_raises(error)), model="res.partner", method="read", args_json="[[999]]")
    assert "Field name error" not in (response.hint or "")
    assert "unknown or was deleted" in response.suggestion


def test_an_invalid_field_in_search_read_is_reported_once_with_the_field_hint():
    client = _client(_raises(INVALID_FIELD_500))
    response = _call(client, model="res.partner", method="search_read", kwargs_json='{"domain": [["nope", "=", 1]]}')

    assert response.success is False
    assert client.execute_method.call_count == 1  # no search + read retry
    assert "Fallback also failed" not in response.error
    assert "odoo://model/res.partner/fields" in response.hint


def test_another_search_read_500_still_falls_back():
    def _execute(model, method, *args, **kwargs):
        if method == "search_read":
            raise ValueError("Request failed: 500 INTERNAL SERVER ERROR\nOdoo error: NotImplemented")
        return [7] if method == "search" else [{"id": 7}]

    response = _call(_client(_execute), model="stock.move.line", method="search_read", kwargs_json='{"domain": []}')
    assert response.success is True
    assert response.fallback_used is True


# ----- blocked is final, not pending -----


def test_a_blocked_call_is_not_pending_confirmation():
    client = _client()
    response = _call(client, model="res.users", method="write", args_json='[[2], {"name": "x"}]')

    assert response.success is False
    assert response.safety.risk_level.value == "blocked"
    assert response.pending_confirmation is False
    assert "confirmation_token" not in (response.hint or "")
    client.execute_method.assert_not_called()


def test_a_blocked_batch_is_not_pending_confirmation():
    client = _client()
    response = _batch(client, [{"model": "res.users", "method": "write", "args_json": '[[2], {"name": "x"}]'}])

    assert response.overall_risk == "blocked"
    assert response.pending_confirmation is False
    client.execute_method.assert_not_called()


def test_a_gated_call_is_still_pending_confirmation():
    response = _call(_client(), model="res.partner", method="write", args_json='[[7], {"name": "x"}]')
    assert response.pending_confirmation is True
    assert "confirmation_token" in response.hint
