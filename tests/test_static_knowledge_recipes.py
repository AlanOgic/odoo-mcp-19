"""Static recipes must not send an agent down a path Odoo refuses or that misleads it.

Each fact was read in the Odoo 19.0 source (sale, purchase, hr, mail, rpc/controllers/json2.py)
and checked on Odoo Online 19.3 on 2026-11-26: states, method names, ``message_post`` with
``mail.mt_note`` read back, and the invoicing wizard. Creating ``mail.message`` notifies nobody.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from odoo_mcp import orm_guides, resources
from odoo_mcp.constants import _DEFAULT_BOOTSTRAP_MODELS, MODEL_STATE_MACHINES, TOOL_REGISTRY, _validate_model

PACKAGE = Path(resources.__file__).resolve().parent
KNOWLEDGE = json.loads((PACKAGE / "module_knowledge.json").read_text())


def _transition(model, method):
    return next(t for t in MODEL_STATE_MACHINES[model]["transitions"] if t["method"] == method)


# ----- registry entries: a `method` means one execute_method call -----


def test_no_registry_recipe_calls_a_private_method():
    methods = {name: entry["method"] for name, entry in TOOL_REGISTRY.items() if "method" in entry}
    methods.update(
        {
            f"{name}[{i}]": step["method"]
            for name, e in TOOL_REGISTRY.items()
            for i, step in enumerate(e.get("steps", []))
        }
    )
    assert {name: method for name, method in methods.items() if method.startswith("_")} == {}


def test_a_registry_entry_with_a_method_names_a_callable_model():
    """odoo://tool-registry says such an entry maps to execute_method(model, method)."""
    for name, entry in TOOL_REGISTRY.items():
        if "method" in entry:
            assert _validate_model(entry["model"]) is None, name


def test_no_module_special_method_or_workflow_step_is_private():
    for module, info in KNOWLEDGE["modules"].items():
        assert [m for m in info.get("special_methods", {}) if m.startswith("_")] == [], module
        for steps in info.get("workflows", {}).values():
            assert [s for s in steps if s.rsplit(".", 1)[-1].startswith("_")] == [], module


# ----- invoicing goes through the wizard -----


def test_invoicing_recipe_is_two_wizard_calls():
    recipe = TOOL_REGISTRY["create_invoice_from_order"]
    assert "method" not in recipe  # two calls: not a single execute_method
    create, run = recipe["steps"]
    assert (create["model"], create["method"]) == ("sale.advance.payment.inv", "create")
    assert create["vals_list"][0]["sale_order_ids"][0][:2] == [6, 0]
    assert (run["model"], run["method"]) == ("sale.advance.payment.inv", "create_invoices")
    assert {"amount", "fixed_amount"} <= set(recipe["params"])
    assert "invoice_ids" in recipe["note"]
    assert "sale.advance.payment.inv" in KNOWLEDGE["modules"]["sale"]["notes"]


def test_the_invoicing_recipe_is_listed_with_its_steps_for_sale_order():
    client = MagicMock()
    client.search_read.return_value = []
    with patch.object(resources, "get_odoo_client", return_value=client):
        payload = json.loads(resources.discover_actions_resource("sale.order"))
    recipe = next(w for w in payload["workflows"] if w["name"] == "create_invoice_from_order")
    assert [step["method"] for step in recipe["steps"]] == ["create", "create_invoices"]


def test_the_shipped_package_never_offers_create_invoices_as_a_call():
    text = "\n".join(p.read_text() for p in PACKAGE.glob("*.py"))
    assert "method: _create_invoices" not in text
    assert '"method": "_create_invoices"' not in text


# ----- sale.order / purchase.order: no 'done' state in Odoo 19, locking is a boolean -----


@pytest.mark.parametrize(
    "model,lock,unlock",
    [("sale.order", "action_lock", "action_unlock"), ("purchase.order", "button_lock", "button_unlock")],
)
def test_orders_have_no_done_state_and_lock_through_a_flag(model, lock, unlock):
    machine = MODEL_STATE_MACHINES[model]
    assert "done" not in machine["states"]
    for transition in machine["transitions"]:
        assert transition["to"] in machine["states"], transition
        sources = transition["from"] if isinstance(transition["from"], list) else [transition["from"]]
        assert set(sources) <= set(machine["states"]), transition
        assert transition["method"] not in (lock, unlock)
    (flag,) = machine["flags"]
    assert (flag["field"], flag["set"], flag["unset"]) == ("locked", lock, unlock)


@pytest.mark.parametrize("model,cancel", [("sale.order", "action_cancel"), ("purchase.order", "button_cancel")])
def test_cancel_says_a_locked_order_refuses_it(model, cancel):
    assert "locked" in _transition(model, cancel)["note"]


def test_sale_order_can_go_back_to_draft():
    assert _transition("sale.order", "action_draft")["from"] == ["sent", "cancel"]


def test_purchase_order_has_the_approval_step_and_no_send_transition():
    machine = MODEL_STATE_MACHINES["purchase.order"]
    assert "to approve" in machine["states"]
    approve = _transition("purchase.order", "button_approve")
    assert (approve["from"], approve["to"]) == ("to approve", "purchase")
    assert "read the state back" in approve["note"]
    # action_rfq_send only opens the e-mail composer: it is not a state transition.
    assert "action_rfq_send" not in [t["method"] for t in machine["transitions"]]
    assert "action_rfq_send" in machine["notes"][0]
    assert _transition("purchase.order", "button_draft")["to"] == "draft"


# ----- hr.contract does not exist in Odoo 19 -----


def test_hr_contract_is_replaced_by_hr_version():
    assert "hr.contract" not in KNOWLEDGE["model_to_module"]
    assert KNOWLEDGE["model_to_module"]["hr.version"] == "hr"


# ----- chatter: message_post, never a hand-made mail.message -----


def test_send_message_recipe_posts_with_message_post_then_reads_back():
    recipe = TOOL_REGISTRY["send_message"]
    assert "method" not in recipe and recipe["model"] == "mail.thread"
    post, verify = recipe["steps"]
    assert post["method"] == "message_post"
    assert post["kwargs"]["body_is_html"] is True and post["kwargs"]["subtype_xmlid"] == "mail.mt_note"
    assert "subtype_id" not in post["kwargs"] and "subtype_id" not in recipe["params"]
    assert (verify["model"], verify["method"]) == ("mail.message", "read")
    assert "notified_partner_ids" in verify["fields"]


def test_the_chatter_recipe_is_found_by_searching_mail():
    payload = json.loads(resources.search_tools_resource("mail"))
    assert "send_message" in [w["name"] for w in payload["workflows"]]


def test_no_shipped_text_tells_agents_to_create_mail_message():
    text = "\n".join(p.read_text() for p in [*PACKAGE.glob("*.py"), PACKAGE / "module_knowledge.json"])
    for stale in ("Create mail.message directly", "fails via JSON-2", "does not work via JSON-2"):
        assert stale not in text, stale


def test_mail_knowledge_names_subtypes_by_xmlid():
    mail = KNOWLEDGE["modules"]["mail"]
    assert "subtype_id" not in mail["field_values"]
    assert set(mail["field_values"]["subtype_xmlid"]) >= {"mail.mt_note", "mail.mt_comment"}
    for example in mail["post_example"].values():
        assert example["method"] == "message_post" and example["kwargs"]["body_is_html"] is True
        # message_post refuses 'model' and 'res_id' as keyword arguments.
        assert not {"model", "res_id", "subtype_id"} & set(example["kwargs"])
    assert "_warning" in mail["post_example"]["message_to_followers"]


def test_mail_thread_guide_says_who_is_notified():
    who = orm_guides.mail_thread_reference()["who_is_notified"]
    assert "partner_ids" in who["rule"] and "does not change the recipients" in who["rule"]
    assert (who["internal_note"]["message_type"], who["internal_note"]["subtype_xmlid"]) == ("comment", "mail.mt_note")
    assert (who["system_line"]["message_type"], who["system_line"]["subtype_xmlid"]) == ("notification", "mail.mt_note")
    # Same recipients as a note: a system line with partner_ids does notify them.
    assert "unless partner_ids are passed" in who["system_line"]["notifies"]
    followers = who["message_to_followers"]
    assert (followers["message_type"], followers["subtype_xmlid"]) == ("comment", "mail.mt_comment")
    assert "approval" in followers["notifies"]
    assert "notified_partner_ids" in who["verify"]
    post = orm_guides.mail_thread_reference()["message_post"]
    assert post["json2_example"]["body_is_html"] is True
    assert "internal API user" in post["params"]["body_is_html"]


# ----- session-bootstrap: an empty variable means "use the default" -----


@pytest.mark.parametrize("value", ["", None])
def test_bootstrap_uses_the_default_models_when_the_variable_is_empty_or_unset(monkeypatch, value):
    """docker-compose forwards an unset MCP_BOOTSTRAP_MODELS as an empty string."""
    if value is None:
        monkeypatch.delenv("MCP_BOOTSTRAP_MODELS", raising=False)
    else:
        monkeypatch.setenv("MCP_BOOTSTRAP_MODELS", value)
    fields = {"name": {"type": "char", "string": "Name"}}
    with patch.object(resources, "_fetch_model_fields", return_value=(fields, None)):
        payload = json.loads(resources.get_session_bootstrap())
    assert sorted(payload["schemas"]) == sorted(_DEFAULT_BOOTSTRAP_MODELS.split(","))


# ----- `confirmed` alone does not bypass the gate, and the tools must not say it does -----


def test_tool_docstrings_do_not_promise_a_bypass():
    """The docstring is the parameter description every client loads."""
    source = (PACKAGE / "server.py").read_text()
    assert "bypass safety confirmation" not in source
    assert "together with confirmation_token" in source
