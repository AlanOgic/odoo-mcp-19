"""Characterization tests for the execution paths of the three write-capable tools.

Before the v1.17 refactor these branches had no direct unit test: the
``resolve_json`` Many2one resolution, the ``search_read`` → ``search`` + ``read``
fallback, the search defaults, both ``execute_workflow`` bodies and
``batch_execute`` with ``atomic=False``. They pin the observable behaviour so the
long functions can be split into helpers without drift.

No live Odoo: ``get_odoo_client`` is patched with a stub. Tests that exercise a
MEDIUM-risk method run in ``permissive`` mode so the token gate does not get in
the way — the gate itself is pinned by ``test_token_gate.py``.
"""

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import odoo_mcp.server as server
from odoo_mcp.constants import DEFAULT_LIMIT, MAX_LIMIT


@pytest.fixture(autouse=True)
def _permissive(monkeypatch):
    monkeypatch.delenv("MCP_READ_ONLY", raising=False)
    monkeypatch.delenv("MCP_WRITE_ALLOWLIST", raising=False)
    monkeypatch.delenv("MCP_VALIDATE_PAYLOADS", raising=False)
    monkeypatch.setenv("MCP_SAFETY_MODE", "permissive")


def _client(**side_effects):
    client = MagicMock()
    for name, effect in side_effects.items():
        getattr(client, name).side_effect = effect
    return client


def _call(client, **kwargs):
    with patch.object(server, "get_odoo_client", return_value=client):
        return server.execute_method(ctx=MagicMock(), **kwargs)


def _sent(client):
    """(model, method, args, kwargs) of every execute_method call on the stub."""
    return [(c.args[0], c.args[1], list(c.args[2:]), dict(c.kwargs)) for c in client.execute_method.call_args_list]


# ----- resolve_json -----


def _resolver(matches, executed=True):
    def _execute(model, method, *args, **kwargs):
        if method == "name_search":
            return matches
        return executed

    return _client(execute_method=_execute)


def test_resolve_json_injects_id_into_write_vals():
    client = _resolver([[42, "Alice"]])
    response = _call(
        client,
        model="res.partner",
        method="write",
        args_json='[[1], {"name": "X"}]',
        resolve_json='{"country_id": {"model": "res.country", "search": "Belgium"}}',
    )
    assert response.success is True
    assert ("res.country", "name_search", [], {"name": "Belgium", "limit": 5}) in _sent(client)
    assert ("res.partner", "write", [[1], {"name": "X", "country_id": 42}], {}) in _sent(client)


def test_resolve_json_injects_id_into_create_vals_dict_and_list():
    client = _resolver([[42, "Alice"]])
    single = _call(
        client,
        model="res.partner",
        method="create",
        args_json='[{"name": "X"}]',
        resolve_json='{"country_id": {"model": "res.country", "search": "Belgium"}}',
    )
    batch = _call(
        client,
        model="res.partner",
        method="create",
        args_json='[[{"name": "X"}, {"name": "Y"}]]',
        resolve_json='{"country_id": {"model": "res.country", "search": "Belgium"}}',
    )
    assert single.success and batch.success
    creates = [call for call in _sent(client) if call[1] == "create"]
    assert creates[0][2] == [{"name": "X", "country_id": 42}]
    assert creates[1][2] == [[{"name": "X", "country_id": 42}, {"name": "Y", "country_id": 42}]]


def test_resolve_json_no_match_reports_error_with_hint():
    client = _resolver([])
    response = _call(
        client,
        model="res.partner",
        method="write",
        args_json="[[1], {}]",
        resolve_json='{"country_id": {"model": "res.country", "search": "Nowhere"}}',
    )
    assert response.success is False
    assert "No match for 'Nowhere' in res.country" in response.error
    assert "res.country" in response.hint
    assert not [c for c in _sent(client) if c[1] == "write"]


def test_resolve_json_ambiguous_match_lists_options():
    client = _resolver([[1, "Belgium"], [2, "Belgium (old)"]])
    response = _call(
        client,
        model="res.partner",
        method="write",
        args_json="[[1], {}]",
        resolve_json='{"country_id": {"model": "res.country", "search": "Belgium"}}',
    )
    assert response.success is False
    assert "Ambiguous match" in response.error
    assert "1: Belgium" in response.hint and "2: Belgium (old)" in response.hint


@pytest.mark.parametrize(
    "spec, fragment",
    [
        ('{"country_id": {"model": "res.country"}}', "requires 'model' and 'search'"),
        ('{"country_id": {"model": "Not A Model", "search": "x"}}', "resolve_json['country_id']"),
        ('{"user_id": {"model": "res.users", "search": "x"}}', "blocked for safety"),
        ('{"country_id": 5}', "requires 'model' and 'search'"),
        ("[1, 2]", "must be a JSON object"),
        ("{not json", "Invalid resolve_json"),
    ],
)
def test_resolve_json_rejects_bad_specs_before_any_call(spec, fragment):
    client = _resolver([[42, "x"]])
    response = _call(client, model="res.partner", method="write", args_json="[[1], {}]", resolve_json=spec)
    assert response.success is False
    assert fragment in response.error
    assert client.execute_method.call_count == 0


def test_resolve_json_malformed_name_search_result_is_reported_as_resolve_failure():
    """A broken name_search tuple must surface as a resolve_json error, not a generic failure."""
    client = _resolver([[1], [2]])  # ambiguous, and each hit lacks its display name
    response = _call(
        client,
        model="res.partner",
        method="write",
        args_json="[[1], {}]",
        resolve_json='{"country_id": {"model": "res.country", "search": "Belgium"}}',
    )
    assert response.success is False
    assert response.error.startswith("resolve_json: Failed to resolve 'country_id'")
    assert not [c for c in _sent(client) if c[1] == "write"]


# ----- search defaults and domain normalisation -----


def test_search_read_gets_default_limit_and_cap():
    client = _client(execute_method=lambda *a, **k: [])
    _call(client, model="res.partner", method="search_read", kwargs_json='{"domain": []}')
    _call(client, model="res.partner", method="search_read", kwargs_json='{"domain": [], "limit": 5000}')
    limits = [call[3]["limit"] for call in _sent(client)]
    assert limits == [DEFAULT_LIMIT, MAX_LIMIT]


def test_double_wrapped_domain_is_unwrapped():
    client = _client(execute_method=lambda *a, **k: 3)
    _call(client, model="res.partner", method="search_count", args_json='[[[["name", "=", "x"]]]]')
    assert _sent(client)[0][2] == [[["name", "=", "x"]]]


# ----- search_read fallback -----


def _failing_search_read(records):
    def _execute(model, method, *args, **kwargs):
        if method == "search_read":
            raise ValueError("Request failed: 500 Internal Server Error for url: x")
        if method == "search":
            return [r["id"] for r in records]
        if method == "read":
            return records
        raise AssertionError(method)

    return _client(execute_method=_execute)


def test_search_read_500_falls_back_to_search_plus_read():
    client = _failing_search_read([{"id": 1, "name": "A"}])
    response = _call(
        client,
        model="stock.move.line",
        method="search_read",
        kwargs_json='{"domain": [["picking_type_id", "!=", false]], "fields": ["name"], "limit": 10, "order": "id"}',
    )
    assert response.success is True
    assert response.fallback_used is True
    assert response.result == [{"id": 1, "name": "A"}]
    assert response.issue_analysis is not None
    assert response.note.startswith("Fallback search+read used")
    sent = _sent(client)
    assert (
        "stock.move.line",
        "search",
        [],
        {"domain": [["picking_type_id", "!=", False]], "limit": 10, "offset": 0, "order": "id"},
    ) in sent
    assert ("stock.move.line", "read", [[1]], {"fields": ["name"]}) in sent


def test_search_read_fallback_with_no_ids_skips_read():
    client = _failing_search_read([])
    response = _call(client, model="res.partner", method="search_read", kwargs_json='{"domain": []}')
    assert response.success is True and response.result == []
    assert not [c for c in _sent(client) if c[1] == "read"]


def test_search_read_fallback_failure_reports_both_errors():
    def _execute(model, method, *args, **kwargs):
        if method == "search_read":
            raise ValueError("500 Internal Server Error")
        raise ValueError("search also broke")

    response = _call(_client(execute_method=_execute), model="res.partner", method="search_read", kwargs_json="{}")
    assert response.success is False
    assert "500" in response.error and "search also broke" in response.error
    assert "model-limitations" in response.suggestion


def test_field_error_gets_schema_hint():
    def _execute(*a, **k):
        raise ValueError("Invalid field 'foo' on model 'res.partner'")

    response = _call(_client(execute_method=_execute), model="res.partner", method="search_read", kwargs_json="{}")
    assert response.success is False
    assert "odoo://model/res.partner/fields" in response.hint


# ----- batch_execute -----


def _batch(client, **kwargs):
    with patch.object(server, "get_odoo_client", return_value=client):
        return asyncio.run(server.batch_execute(progress=AsyncMock(), **kwargs))


def _flaky(fail_index):
    calls = {"n": 0}

    def _execute(model, method, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] - 1 == fail_index:
            raise ValueError("boom")
        return {"ok": calls["n"]}

    return _client(execute_method=_execute)


_OPS = [
    {"model": "res.partner", "method": "search_read", "kwargs_json": "{}"},
    {"model": "res.partner", "method": "search_count", "kwargs_json": "{}"},
    {"model": "res.partner", "method": "read", "args_json": "[[1]]"},
]


def test_batch_non_atomic_continues_past_failure():
    response = _batch(_flaky(1), operations=_OPS, atomic=False)
    assert response.success is False
    assert [r.success for r in response.results] == [True, False, True]
    assert response.failed_operations == 1 and response.successful_operations == 2
    assert response.error == "1 operations failed"


def test_batch_atomic_stops_at_first_failure_and_keeps_completed_results():
    response = _batch(_flaky(1), operations=_OPS, atomic=True)
    assert response.success is False
    assert [r.success for r in response.results] == [True, False]
    assert response.error.startswith("Failed at operation 1")


def test_batch_reports_invalid_operation_without_calling_odoo():
    client = _client(execute_method=lambda *a, **k: 1)
    response = _batch(client, operations=[{"model": "res.partner", "method": "read", "args_json": "{not json"}])
    assert response.results[0].success is False
    assert response.results[0].error  # the JSON decode message
    assert client.execute_method.call_count == 0


# ----- execute_workflow -----


def _workflow(client, **kwargs):
    """Run a workflow, confirming through the token gate when it asks."""

    async def _run():
        response = await server.execute_workflow(progress=AsyncMock(), **kwargs)
        if response.pending_confirmation:
            token = re.search(r"confirmation_token='([^']+)'", response.tip).group(1)
            response = await server.execute_workflow(
                progress=AsyncMock(), confirmed=True, confirmation_token=token, **kwargs
            )
        return response

    with patch.object(server, "get_odoo_client", return_value=client):
        return asyncio.run(_run())


def test_lead_to_won_converts_a_lead_then_marks_it_won():
    client = _client(search_read=lambda *a, **k: [{"id": 7, "type": "lead"}], execute_method=lambda *a, **k: True)
    response = _workflow(client, workflow="lead_to_won", params_json='{"lead_id": 7, "partner_id": 3}')
    assert response.success is True
    assert [s.step for s in response.steps] == ["convert_to_opportunity", "mark_won"]
    assert ("crm.lead", "convert_opportunity", [[7]], {"partner_id": 3}) in _sent(client)
    assert ("crm.lead", "action_set_won", [[7]], {}) in _sent(client)


def test_lead_to_won_skips_conversion_for_an_opportunity():
    client = _client(
        search_read=lambda *a, **k: [{"id": 7, "type": "opportunity"}], execute_method=lambda *a, **k: True
    )
    response = _workflow(client, workflow="crm_workflow", params_json='{"lead_id": 7}')
    assert response.success is True
    assert response.steps[0].skipped is True
    assert [c[1] for c in _sent(client)] == ["action_set_won"]


def test_lead_to_won_reports_a_failed_step():
    def _execute(model, method, *a, **k):
        raise ValueError("cannot win")

    client = _client(search_read=lambda *a, **k: [{"id": 7, "type": "opportunity"}], execute_method=_execute)
    response = _workflow(client, workflow="lead_to_won", params_json='{"lead_id": 7}')
    assert response.success is False
    assert response.steps[1].success is False and "cannot win" in response.steps[1].error


def test_lead_to_won_requires_lead_id():
    response = _workflow(_client(), workflow="lead_to_won", params_json="{}")
    assert response.success is False and "lead_id" in response.error


def test_create_and_post_invoice_creates_then_posts():
    def _execute(model, method, *a, **k):
        return 99 if method == "create" else True

    client = _client(execute_method=_execute)
    response = _workflow(
        client,
        workflow="quick_invoice",
        params_json='{"partner_id": 5, "lines": [{"product_id": 1, "quantity": 2, "price_unit": 10}]}',
    )
    assert response.success is True and response.invoice_id == 99
    create = next(c for c in _sent(client) if c[1] == "create")
    vals = create[2][0][0]  # args == ([invoice_vals],)
    assert vals["move_type"] == "out_invoice" and vals["partner_id"] == 5
    assert vals["invoice_line_ids"] == [(0, 0, {"product_id": 1, "quantity": 2, "price_unit": 10, "name": "Product"})]
    assert ("account.move", "action_post", [[99]], {}) in _sent(client)


def test_create_and_post_invoice_can_skip_posting():
    client = _client(execute_method=lambda model, method, *a, **k: 99)
    response = _workflow(
        client,
        workflow="create_and_post_invoice",
        params_json='{"partner_id": 5, "lines": [{"product_id": 1}], "post": false}',
    )
    assert response.success is True
    assert [c[1] for c in _sent(client)] == ["create"]


def test_create_and_post_invoice_stops_when_create_fails():
    def _execute(*a, **k):
        raise ValueError("no journal")

    response = _workflow(
        _client(execute_method=_execute),
        workflow="create_and_post_invoice",
        params_json='{"partner_id": 5, "lines": [{"product_id": 1}]}',
    )
    assert response.success is False
    assert [s.step for s in response.steps] == ["create_invoice"]
    assert "no journal" in response.steps[0].error


@pytest.mark.parametrize(
    "params, fragment",
    [('{"lines": [{"product_id": 1}]}', "partner_id required"), ('{"partner_id": 5}', "lines required")],
)
def test_create_and_post_invoice_validates_params(params, fragment):
    response = _workflow(_client(), workflow="create_and_post_invoice", params_json=params)
    assert response.success is False and fragment in response.error


def test_unknown_workflow_lists_available_ones():
    response = _workflow(_client(), workflow="stock_transfer", params_json="{}")
    assert response.success is False
    assert response.error == "Unknown workflow: stock_transfer"
    assert any(w.startswith("lead_to_won") for w in response.available_workflows)


def test_invalid_params_json_is_rejected():
    response = _workflow(_client(), workflow="lead_to_won", params_json="{oops")
    assert response.success is False and "Invalid params_json" in response.error


# ----- batch_execute: per-operation parsing (shared with execute_method) -----


def test_batch_merges_default_context_into_each_operation(monkeypatch):
    monkeypatch.setenv("MCP_DEFAULT_CONTEXT", '{"lang": "fr_FR"}')
    client = _client(execute_method=lambda *a, **k: [])
    ops = [{"model": "res.partner", "method": "search_read", "kwargs_json": '{"context": {"tz": "UTC"}}'}]
    response = _batch(client, operations=ops)
    assert response.success is True
    assert _sent(client)[0][3]["context"] == {"lang": "fr_FR", "tz": "UTC"}


def test_batch_non_list_args_json_reports_the_operation_index():
    client = _client(execute_method=lambda *a, **k: 1)
    response = _batch(client, operations=[{"model": "res.partner", "method": "read", "args_json": '{"a": 1}'}])
    assert response.results[0].success is False
    assert response.results[0].error.startswith("Operation 0:")
    assert "JSON array" in response.results[0].error
    assert client.execute_method.call_count == 0
