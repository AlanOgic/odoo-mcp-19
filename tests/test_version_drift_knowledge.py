"""Static knowledge must not steer agents to methods Odoo has removed.

Checked live against Odoo Online 19.3 (``/doc-bearer``) on 2026-09-30: every name
below answered 404 "The method ... does not exist".
"""

import json
from importlib import resources
from unittest.mock import MagicMock

import pytest

from odoo_mcp.constants import ERROR_CATEGORIES, PRIVATE_METHOD_HINTS
from odoo_mcp.utils import get_error_suggestion

KNOWLEDGE = json.loads(resources.files("odoo_mcp").joinpath("module_knowledge.json").read_text())

MISSING_METHOD_ERROR = (
    "Request failed: 404 NOT FOUND for url: https://x/json/2/res.partner/read_group\n"
    "Odoo error: The method 'res.partner.read_group' does not exist"
)


@pytest.mark.parametrize(
    ("module", "method"),
    [
        ("account", "button_set_checked"),
        ("stock", "button_scrap"),
        ("hr_expense", "action_submit_expenses"),
        ("hr_expense", "action_approve_expense_sheets"),
        ("hr_expense", "action_refuse_expense"),
        ("hr_leave", "action_confirm"),
        ("project", "action_assign_to_me"),
        ("documents", "document_create"),
        ("discuss", "channel_create"),
        ("knowledge", "article_duplicate"),
        ("ai", "get_direct_response"),
        ("ai", "create_from_attachments"),
    ],
)
def test_nonexistent_special_method_is_not_documented(module, method):
    assert method not in KNOWLEDGE["modules"][module]["special_methods"]


@pytest.mark.parametrize(
    ("module", "method"),
    [
        ("account", "set_moves_checked"),
        ("stock", "action_scrap"),
        ("hr_expense", "action_submit"),
        ("hr_expense", "action_approve"),
        ("hr_expense", "action_refuse"),
    ],
)
def test_replacement_special_method_is_documented(module, method):
    assert method in KNOWLEDGE["modules"][module]["special_methods"]


def test_orm_signatures_match_the_json2_parameter_names():
    orm = KNOWLEDGE["orm_methods"]

    assert "count" not in orm["search"]["signature"]
    assert orm["default_get"]["signature"] == "Model.default_get(fields)"
    assert "fields" in orm["default_get"]["parameters"]
    assert "fields_list" not in orm["default_get"]["parameters"]


def test_read_group_is_documented_as_19_0_only():
    status = KNOWLEDGE["aggregation"]["read_group"]["status"]

    assert "19.0 only" in status
    assert "formatted_read_group" in status


def test_removed_domain_operators_are_documented():
    removed = KNOWLEDGE["domain_syntax"]["removed_operators"]

    assert removed["<>"] == "use !="
    assert removed["=="] == "use ="


def test_hints_never_recommend_read_group():
    assert "read_group (deprecated" not in PRIVATE_METHOD_HINTS["_read_group"]
    for category in ERROR_CATEGORIES.values():
        assert "Use read_group" not in " ".join(category["solutions"])


def test_missing_method_404_is_not_reported_as_a_wrong_model():
    suggestion = get_error_suggestion(MISSING_METHOD_ERROR, "res.partner", "read_group")

    assert "Method not found" in suggestion
    assert "odoo://api/version-drift" in suggestion
    assert "Verify the model name" not in suggestion


def test_missing_method_hint_is_not_a_field_name_error(monkeypatch):
    odoo = MagicMock()
    odoo.execute_method.side_effect = RuntimeError(MISSING_METHOD_ERROR)
    monkeypatch.setattr("odoo_mcp.server.get_odoo_client", lambda: odoo)
    from odoo_mcp.server import execute_method

    response = execute_method(ctx=MagicMock(), model="res.partner", method="read_group", kwargs_json='{"domain": []}')

    assert response.success is False
    assert "Field name error" not in response.hint
    assert "odoo://methods/res.partner" in response.hint
