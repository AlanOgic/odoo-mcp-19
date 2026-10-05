"""batch_execute's confirmation gate, driven through the tool.

``tests/test_token_gate.py`` pins the token primitives; these tests pin what the
tool does with them now that ``batch_execute`` is the only multi-step path:
BLOCKED operations are refused before any token exists, a token is bound to the
exact operations list and to nothing a role change could reuse, and a confirmed
re-call with the right token runs every operation.
"""

import asyncio
import json
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from odoo_mcp import server

WRITE_OPS = [
    {"model": "res.partner", "method": "write", "args_json": '[[7], {"name": "Renamed"}]'},
    {"model": "res.partner", "method": "write", "args_json": '[[8], {"name": "Renamed too"}]'},
]


@pytest.fixture
def odoo(monkeypatch):
    client = MagicMock()
    client.execute_method.return_value = True
    monkeypatch.setattr(server, "get_odoo_client", lambda: client)
    return client


@pytest.fixture
def role(monkeypatch):
    current = {"value": None}  # None: STDIO / single-user operator
    monkeypatch.setattr(server, "current_role", lambda: current["value"])
    return current


def _batch(operations, **kwargs):
    return asyncio.run(server.batch_execute(operations=operations, progress=AsyncMock(), **kwargs))


def _token_in(response) -> str:
    match = re.search(r"confirmation_token='([^']+)'", response.error or "")
    assert match, f"no confirmation token offered: {response.error!r}"
    return match.group(1)


class TestBlockedOperationsGetNoToken:
    def test_a_readonly_caller_is_refused_before_any_token(self, odoo, role):
        role["value"] = "readonly"

        response = _batch(WRITE_OPS)

        assert response.success is False
        assert response.overall_risk == "blocked"
        assert "confirmation_token" not in (response.error or "")
        odoo.execute_method.assert_not_called()

    def test_a_token_issued_to_another_role_does_not_unlock_a_readonly_caller(self, odoo, role):
        token = _token_in(_batch(WRITE_OPS))

        role["value"] = "readonly"
        response = _batch(WRITE_OPS, confirmed=True, confirmation_token=token)

        assert response.success is False
        assert response.overall_risk == "blocked"
        odoo.execute_method.assert_not_called()

    def test_an_operation_outside_the_write_allowlist_is_refused(self, monkeypatch, odoo, role):
        monkeypatch.setenv("MCP_WRITE_ALLOWLIST", "sale.order.action_confirm")

        response = _batch(WRITE_OPS)

        assert response.success is False
        assert response.overall_risk == "blocked"
        assert "confirmation_token" not in (response.error or "")
        odoo.execute_method.assert_not_called()


class TestTheConfirmationRoundTrip:
    def test_the_first_call_offers_a_token_and_runs_nothing(self, odoo, role):
        response = _batch(WRITE_OPS)

        assert response.success is False
        assert response.pending_confirmation is True
        assert _token_in(response)
        odoo.execute_method.assert_not_called()

    def test_the_confirmed_re_call_with_the_token_runs_every_operation(self, odoo, role):
        token = _token_in(_batch(WRITE_OPS))

        response = _batch(WRITE_OPS, confirmed=True, confirmation_token=token)

        assert response.success is True
        assert [call.args[:2] for call in odoo.execute_method.call_args_list] == [
            ("res.partner", "write"),
            ("res.partner", "write"),
        ]

    def test_a_token_does_not_cover_a_substituted_operations_list(self, odoo, role):
        token = _token_in(_batch(WRITE_OPS))
        substituted = json.loads(json.dumps(WRITE_OPS))
        substituted[1]["args_json"] = '[[8, 9, 10], {"name": "Renamed too"}]'

        response = _batch(substituted, confirmed=True, confirmation_token=token)

        assert response.success is False
        odoo.execute_method.assert_not_called()

    def test_confirmed_without_a_token_runs_nothing(self, odoo, role):
        response = _batch(WRITE_OPS, confirmed=True)

        assert response.success is False
        odoo.execute_method.assert_not_called()

    def test_a_token_is_single_use(self, odoo, role):
        token = _token_in(_batch(WRITE_OPS))
        _batch(WRITE_OPS, confirmed=True, confirmation_token=token)
        odoo.execute_method.reset_mock()

        response = _batch(WRITE_OPS, confirmed=True, confirmation_token=token)

        assert response.success is False
        odoo.execute_method.assert_not_called()
