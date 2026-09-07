"""Pin ``odoo://methods/{model}`` against the JSON-2 argument contract.

The static method table published by the resource used to be a hand-copied
list that drifted from ``arg_mapping.V2_ARG_MAPPING`` (``default_get`` said
``fields_list``, ``copy`` said ``id``) — an agent without ``/doc-bearer/``
read the wrong names, sent them, and got a 422. The catalog is now *derived*
from the mapping, and the live enrichment shared by the static and the
discovered methods is one helper.

No live Odoo: ``_get_live_doc`` is patched where a resource is exercised.
"""

import json
from unittest.mock import patch

import pytest

import odoo_mcp.resources as resources
from odoo_mcp import method_catalog
from odoo_mcp.arg_mapping import V2_ARG_MAPPING

# A /doc-bearer/ excerpt shaped like the api_doc module output.
_LIVE_DOC = {
    "methods": {
        "search": {
            "signature": "(domain, offset=0, limit=None, order=None) -> list[int]",
            "return": {"annotation": "list[int]"},
            "api": ["model", "readonly"],
            "module": "core",
            "raise": {"AccessError": "if user is <b>not</b> allowed"},
            "parameters": {
                "domain": {"annotation": "DomainType", "doc": "A search <i>domain</i>."},
                "limit": {"annotation": "int | None", "default": None},
                "undocumented": {},
            },
        },
        "action_quotation_send": {
            "doc": "Open the <b>email</b> composer.",
            "signature": "() -> dict",
            "return": {"annotation": "dict"},
            "api": ["public"],
            "module": "sale",
            "parameters": {"ids": {"annotation": "list[int]", "doc": "Records to send"}},
        },
        "action_confirm": {
            "signature": "() -> bool",
            "module": "sale",
            "parameters": {},
        },
    }
}


def _payload(model="res.partner", live_doc=None):
    """res.partner has no module_knowledge entry, so the live fixture cannot collide with special methods."""
    return method_catalog.build_methods_payload(model, live_doc)


def _orm_entries(payload):
    for category in method_catalog.ORM_CATEGORIES:
        yield from payload[category]


# ----- the bug: static params must equal the JSON-2 mapping -----


def test_static_default_get_publishes_the_real_parameter_name():
    with patch.object(resources, "_get_live_doc", return_value=None):
        payload = json.loads(resources.get_methods("res.partner"))
    default_get = next(m for m in payload["introspection_methods"] if m["name"] == "default_get")
    assert default_get["params"] == ["fields"]


def test_static_copy_publishes_the_record_bound_ids_key():
    payload = _payload()
    copy = next(m for m in payload["write_methods"] if m["name"] == "copy")
    assert copy["params"] == ["ids", "default"]


def test_every_static_method_mirrors_v2_arg_mapping():
    payload = _payload()
    for entry in _orm_entries(payload):
        mapping = V2_ARG_MAPPING[entry["name"]]  # KeyError == catalog names a method the mapping lacks
        assert entry["params"] == [name for _, name in sorted(mapping)], entry["name"]


# ----- live enrichment -----


def test_live_doc_enriches_a_static_method():
    payload = _payload(live_doc=_LIVE_DOC)
    search = next(m for m in payload["read_methods"] if m["name"] == "search")
    assert search["signature"].startswith("(domain, offset=0")
    assert search["return_type"] == "list[int]"
    assert search["api"] == ["model", "readonly"]
    assert search["module"] == "core"
    assert search["exceptions"] == {"AccessError": "if user is not allowed"}
    assert search["param_details"]["domain"] == {"type": "DomainType", "description": "A search domain."}
    assert search["param_details"]["limit"] == {"type": "int | None", "default": None}
    assert "undocumented" not in search["param_details"]
    assert payload["_source"].startswith("live")


def test_live_doc_discovers_additional_methods_sorted_by_module_then_name():
    payload = _payload(live_doc=_LIVE_DOC)
    additional = payload["additional_methods"]
    assert [m["name"] for m in additional] == ["action_confirm", "action_quotation_send"]
    send = additional[1]
    assert send["description"] == "Open the email composer."
    assert send["return_type"] == "dict"
    assert send["params"] == ["ids"]


def test_static_and_discovered_methods_share_one_enrichment_shape():
    payload = _payload(live_doc=_LIVE_DOC)
    search = next(m for m in payload["read_methods"] if m["name"] == "search")
    send = next(m for m in payload["additional_methods"] if m["name"] == "action_quotation_send")
    for key in ("signature", "return_type", "api", "module"):
        assert key in search and key in send, key


def test_without_live_doc_source_is_static_and_no_additional_methods():
    payload = _payload(live_doc=None)
    assert payload["_source"].startswith("static")
    assert "additional_methods" not in payload


# ----- module knowledge -----


def test_module_knowledge_special_methods_and_notes_are_included():
    payload = _payload("account.move")
    names = {m["name"] for m in payload["special_methods"]}
    assert "action_post" in names
    assert payload["warnings"], "account.move has notes in module_knowledge.json"


@pytest.mark.parametrize("model", ["res.partner", "sale.order"])
def test_payload_is_json_serializable(model):
    json.dumps(_payload(model, live_doc=_LIVE_DOC))


def test_discovered_methods_list_param_names_but_not_param_details():
    """Discovered methods are numerous (100+ on sale.order); the pre-refactor resource published
    only their parameter *names*. Keeping per-parameter details off them holds the payload size."""
    payload = _payload(live_doc=_LIVE_DOC)
    send = next(m for m in payload["additional_methods"] if m["name"] == "action_quotation_send")
    assert send["params"] == ["ids"]
    assert "param_details" not in send
