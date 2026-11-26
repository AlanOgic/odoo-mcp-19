"""The prompts must recommend calls that run and that answer the question asked.

Every recipe below was executed on Odoo Online 19.3 on 2026-11-26. What they replace:
``inventory-check`` summed real stock with customer, supplier, production and
inventory-adjustment locations; ``ar-aging-report`` counted cancelled invoices and fetched
no due date to age them with; ``aggregation-report`` presented sale.order fields as examples
for any model; ``hierarchical-query`` read ``child_id``, which res.partner does not have.
"""

import asyncio
import json
import re
from pathlib import Path

import pytest

import odoo_mcp.server  # noqa: F401  registers every prompt and resource
from odoo_mcp import resources
from odoo_mcp.app import mcp
from odoo_mcp.constants import MODEL_STATE_MACHINES
from odoo_mcp.server import _RESOURCE_ROUTES

KNOWLEDGE = json.loads((Path(resources.__file__).resolve().parent / "module_knowledge.json").read_text())
INTERNAL_STOCK = '["location_id.usage", "=", "internal"]'


def _render(name, **arguments):
    result = asyncio.run(mcp.render_prompt(name, arguments))
    return "".join(message.content.text for message in result.messages)


def _prompt_names():
    return sorted(p.name for p in asyncio.run(mcp.list_prompts()))


# ----- inventory: real stock only -----


@pytest.mark.parametrize("arguments", [{}, {"product": "RCP"}])
def test_inventory_check_counts_internal_locations_only(arguments):
    text = _render("inventory-check", **arguments)
    assert text.count('execute_method("stock.quant"') == 1
    assert INTERNAL_STOCK in text


# ----- receivables: posted invoices, with a due date to age them -----


def test_ar_aging_only_counts_posted_invoices():
    text = _render("ar-aging-report")
    domains = re.findall(r'"domain": (\[\[.*?\]\]),\s*"(?:groupby|fields)"', text, re.DOTALL)
    assert len(domains) == 2
    for domain in domains:
        assert ["state", "=", "posted"] in json.loads(domain)


def test_ar_aging_reads_the_due_date_it_buckets_on():
    text = _render("ar-aging-report")
    assert '"invoice_date_due"' in text
    assert text.index('"invoice_date_due"') < text.index("90+ days overdue")


# ----- examples must not claim fields the requested model lacks -----


def test_aggregation_examples_are_not_presented_as_the_requested_model():
    text = _render("aggregation-report", model="res.partner")
    assert "Examples for res.partner" not in text
    assert "written for sale.order" in text
    assert "odoo://model/res.partner/quick-schema" in text


def test_hierarchical_query_does_not_assume_a_children_field_name():
    text = _render("hierarchical-query", model="res.partner")
    assert '"fields": ["name", "child_id"]' not in text
    assert "odoo://model/res.partner/quick-schema" in text


# ----- JSON-2 spelling and budget -----


def test_api_reference_writes_commands_and_domains_as_json_lists():
    text = _render("odoo-api-reference")
    assert "[0, 0, {values}]" in text
    assert not re.search(r"\(\d, (?:0|id), ", text)  # no (0, 0, {...}) tuples
    assert '[("' not in text


def test_api_reference_mentions_name_resolution():
    assert "resolve_json" in _render("odoo-api-reference")


def test_no_prompt_sends_agents_to_the_full_schema():
    """/schema is ~190 KB for res.partner; read_resource cuts at 15 000 characters."""
    for name in _prompt_names():
        arguments = {"customer": "Acme"} if name == "customer-360" else {}
        assert not re.search(r"odoo://model/[^/\s]+/schema", _render(name, **arguments)), name


def test_pagination_prompt_has_no_broken_python_loop():
    text = _render("paginated-search")
    assert "kwargs_json=f'" not in text
    assert "`result`" in text


def test_every_uri_a_prompt_cites_has_a_bridge_route():
    def routed(uri):
        concrete = re.sub(r"\{\w+\}", "res.partner", uri.rstrip("."))
        return any(pattern.match(concrete) for pattern, _handler, _params in _RESOURCE_ROUTES)

    cited = set()
    for name in _prompt_names():
        arguments = {"customer": "Acme"} if name == "customer-360" else {}
        cited |= set(re.findall(r"odoo://[A-Za-z0-9_./{}\-]+", _render(name, **arguments)))
    assert cited
    assert sorted(uri for uri in cited if not routed(uri)) == []


# ----- time off: states and methods as Odoo 19 has them -----


def test_time_off_has_no_draft_state_and_no_confirm_method():
    machine = MODEL_STATE_MACHINES["hr.leave"]
    assert sorted(machine["states"]) == ["cancel", "confirm", "refuse", "validate", "validate1"]
    assert "action_confirm" not in {t["method"] for t in machine["transitions"]}


def test_time_off_transitions_only_use_declared_states():
    machine = MODEL_STATE_MACHINES["hr.leave"]
    for transition in machine["transitions"]:
        sources = transition["from"] if isinstance(transition["from"], list) else [transition["from"]]
        assert set(sources) | {transition["to"]} <= set(machine["states"]), transition["method"]


def test_time_off_approval_says_it_may_stop_at_the_first_approval():
    approve = next(t for t in MODEL_STATE_MACHINES["hr.leave"]["transitions"] if t["from"] == "confirm")
    assert approve["method"] == "action_approve"
    assert "validate1" in approve["note"]


def test_time_off_cancel_is_described_as_a_wizard_not_a_transition():
    machine = MODEL_STATE_MACHINES["hr.leave"]
    assert "action_cancel" not in {t["method"] for t in machine["transitions"]}
    assert any("hr.holidays.cancel.leave" in note for note in machine["notes"])


# ----- knowledge entries that named a wrong or missing model -----


def test_action_view_invoice_is_listed_on_orders_not_on_invoices():
    modules = KNOWLEDGE["modules"]
    assert "action_view_invoice" not in modules["account"]["special_methods"]
    assert "action_view_invoice" in modules["sale"]["special_methods"]
    assert "action_view_invoice" in modules["purchase"]["special_methods"]


def test_document_folders_are_documents():
    models = KNOWLEDGE["hierarchical_models"]["common_models"]
    assert "documents.folder" not in models
    assert models["documents.document"]["parent_field"] == "folder_id"
