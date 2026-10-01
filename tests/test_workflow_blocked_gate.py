"""execute_workflow must refuse BLOCKED steps instead of offering a confirmation token.

execute_method and batch_execute refuse a BLOCKED classification outright. The
workflow gate only issued and validated a token, so a ``readonly`` user (or a call
outside MCP_WRITE_ALLOWLIST) could self-confirm and run the workflow's writes.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

INVOICE_PARAMS = '{"partner_id": 7, "lines": [{"product_id": 1, "quantity": 1, "price_unit": 1}]}'


@pytest.fixture
def odoo(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr("odoo_mcp.server.get_odoo_client", lambda: client)
    return client


def _run(workflow: str, params_json: str, **kwargs):
    from odoo_mcp.server import execute_workflow

    return asyncio.run(execute_workflow(workflow=workflow, params_json=params_json, **kwargs))


@pytest.mark.parametrize(
    ("workflow", "params_json"),
    [("quick_invoice", INVOICE_PARAMS), ("lead_to_won", '{"lead_id": 3}')],
)
def test_readonly_role_is_refused_without_a_token(monkeypatch, odoo, workflow, params_json):
    monkeypatch.setattr("odoo_mcp.server.current_role", lambda: "readonly")

    response = _run(workflow, params_json)

    assert response.success is False
    assert response.pending_confirmation is False
    assert response.overall_risk == "blocked"
    assert "confirmation_token" not in (response.tip or "")
    assert "blocked" in (response.error or "").lower()
    odoo.execute_method.assert_not_called()


def test_readonly_role_cannot_confirm_with_a_token_issued_to_another_role(monkeypatch, odoo):
    role = {"value": "admin"}
    monkeypatch.setattr("odoo_mcp.server.current_role", lambda: role["value"])
    issued = _run("quick_invoice", INVOICE_PARAMS)
    token = issued.tip.split("confirmation_token='")[1].split("'")[0]

    role["value"] = "readonly"
    response = _run("quick_invoice", INVOICE_PARAMS, confirmed=True, confirmation_token=token)

    assert response.success is False
    assert response.overall_risk == "blocked"
    odoo.execute_method.assert_not_called()


def test_step_outside_write_allowlist_is_refused(monkeypatch, odoo):
    monkeypatch.setenv("MCP_WRITE_ALLOWLIST", "res.partner.write")

    response = _run("quick_invoice", INVOICE_PARAMS)

    assert response.success is False
    assert response.pending_confirmation is False
    assert response.overall_risk == "blocked"
    odoo.execute_method.assert_not_called()


def test_unrestricted_caller_still_gets_a_confirmation_token(odoo):
    response = _run("quick_invoice", INVOICE_PARAMS)

    assert response.pending_confirmation is True
    assert "confirmation_token='" in response.tip
