"""The default (strict) profile must gate a single-record write and report its classification.

Before this fix a ``write`` / ``create`` on one non-sensitive record executed from a
single tool call with ``pending_confirmation=False`` and ``safety=None`` — the only
thing that stopped a test run from mutating production was Odoo rejecting fake ids.
"""

from unittest.mock import MagicMock, patch

import pytest

import odoo_mcp.server as server


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.delenv("MCP_READ_ONLY", raising=False)
    monkeypatch.delenv("MCP_WRITE_ALLOWLIST", raising=False)
    monkeypatch.delenv("MCP_VALIDATE_PAYLOADS", raising=False)
    monkeypatch.setenv("MCP_SAFETY_MODE", "strict")


def _call(client, **kwargs):
    with patch.object(server, "get_odoo_client", return_value=client):
        return server.execute_method(ctx=MagicMock(), **kwargs)


@pytest.mark.parametrize(
    "model,method,args_json",
    [
        ("res.partner", "write", '[[42], {"name": "x"}]'),
        ("crm.lead", "write", '[[7], {"expected_revenue": 0}]'),
        ("sale.order", "write", '[[15], {"note": "x"}]'),
        ("sale.order", "create", '[{"partner_id": 1}]'),
    ],
)
def test_single_record_write_is_token_gated_and_not_sent(model, method, args_json):
    client = MagicMock()
    resp = _call(client, model=model, method=method, args_json=args_json)
    assert resp.success is False
    assert resp.pending_confirmation is True
    assert resp.safety is not None and resp.safety.requires_confirmation is True
    assert "confirmation_token=" in resp.hint
    assert client.execute_method.call_count == 0


def test_confirmed_single_write_executes_and_reports_its_classification():
    client = MagicMock()
    client.execute_method.return_value = True
    first = _call(client, model="res.partner", method="write", args_json='[[42], {"name": "x"}]')
    token = first.hint.split("confirmation_token='")[1].split("'")[0]
    resp = _call(
        client,
        model="res.partner",
        method="write",
        args_json='[[42], {"name": "x"}]',
        confirmed=True,
        confirmation_token=token,
    )
    assert resp.success is True
    assert resp.safety is not None
    assert resp.safety.risk_level.value == "medium"
    assert client.execute_method.call_count == 1


def test_safe_read_reports_safe_classification_on_success(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "permissive")
    client = MagicMock()
    client.execute_method.return_value = [{"id": 1}]
    resp = _call(client, model="res.partner", method="search_read", kwargs_json='{"domain": [], "fields": ["id"]}')
    assert resp.success is True
    assert resp.safety is not None and resp.safety.risk_level.value == "safe"
