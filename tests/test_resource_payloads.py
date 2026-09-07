"""Round-trip economy of the large ``odoo://`` resources.

* ``odoo://model/{m}/docs`` issued one ``ir.model.fields.selection`` query per
  selection field (up to 20 sequential round-trips); one query grouped in
  Python is enough.
* The big dynamic emitters (``/schema``, ``odoo://model/{m}``, ``odoo://models``,
  ``/methods``, ``/docs``) were pretty-printed with ``indent=2`` — ~25 % more
  characters for the agent to pay for, and an earlier hit on the 15 000-char
  ``read_resource`` truncation. They are emitted compact, like the quick
  views already were.
* ``odoo://model/{m}`` bypassed ``_fetch_model_fields`` — no name validation
  and a fresh full ``fields_get`` on every read while every other schema view
  went through the shared cache.

No live Odoo: ``get_odoo_client`` is patched with a stub.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import odoo_mcp.resources as resources

_FIELDS = {
    "id": {"type": "integer", "string": "ID", "readonly": True},
    "name": {"type": "char", "string": "Name", "required": True},
    "state": {"type": "selection", "string": "Status", "selection": [["draft", "Draft"], ["done", "Done"]]},
}


def _meta(field_id, name, ttype, required=False, help_text=None):
    """One ir.model.fields row as search_read returns it."""
    return {
        "id": field_id,
        "name": name,
        "field_description": name.title(),
        "help": help_text,
        "ttype": ttype,
        "relation": False,
        "required": required,
    }


_FIELDS_META = [
    _meta(11, "state", "selection", required=True),
    _meta(12, "priority", "selection"),
    _meta(13, "kind", "selection"),
    _meta(14, "name", "char", required=True, help_text="Label"),
]

# Deliberately unsorted so the grouping must order by sequence, not arrival.
_SELECTIONS = [
    {"field_id": [11, "Status (x.y)"], "value": "done", "name": "Done", "sequence": 2},
    {"field_id": [12, "Priority (x.y)"], "value": "1", "name": "High", "sequence": 1},
    {"field_id": [11, "Status (x.y)"], "value": "draft", "name": "Draft", "sequence": 1},
    {"field_id": [13, "Kind (x.y)"], "value": "a", "name": "A", "sequence": 1},
]


def _docs_client():
    client = MagicMock()
    by_model = {
        "ir.model": [{"name": "Thing", "info": "A thing", "modules": "x"}],
        "ir.model.fields": _FIELDS_META,
        "ir.model.fields.selection": _SELECTIONS,
        "ir.actions.act_window": [],
    }
    client.search_read.side_effect = lambda model, *a, **k: by_model[model]
    return client


def _selection_calls(client):
    return [c for c in client.search_read.call_args_list if c.args[0] == "ir.model.fields.selection"]


# ----- P1: one query for every selection field -----


def test_model_docs_fetches_all_selection_options_in_one_query():
    client = _docs_client()
    with patch.object(resources, "get_odoo_client", return_value=client):
        payload = json.loads(resources.get_model_docs("x.y"))
    assert len(_selection_calls(client)) == 1
    assert payload["selection_options"]["state"] == [
        {"value": "draft", "label": "Draft"},
        {"value": "done", "label": "Done"},
    ]
    assert payload["selection_options"]["priority"] == [{"value": "1", "label": "High"}]
    assert payload["selection_options"]["kind"] == [{"value": "a", "label": "A"}]
    assert "name" not in payload["selection_options"]


def test_model_docs_skips_the_selection_query_when_no_selection_field():
    client = _docs_client()
    no_selection = [m for m in _FIELDS_META if m["ttype"] != "selection"]
    client.search_read.side_effect = lambda model, *a, **k: (
        no_selection if model == "ir.model.fields" else _docs_client().search_read(model)
    )
    with patch.object(resources, "get_odoo_client", return_value=client):
        payload = json.loads(resources.get_model_docs("x.y"))
    assert _selection_calls(client) == []
    assert payload["selection_options"] == {}


# ----- P2: compact JSON on the large emitters -----


def _schema_client():
    client = MagicMock()
    client.get_model_fields.side_effect = lambda model, attributes=None: _FIELDS
    client.get_model_info.return_value = {"name": "Thing", "model": "x.y"}
    client.get_models.return_value = {"model_names": ["x.y"], "models_details": {"x.y": {"name": "Thing"}}}
    return client


@pytest.mark.parametrize(
    "call",
    [
        lambda: resources.get_model_schema("x.y"),
        lambda: resources.get_model_info("x.y"),
        lambda: resources.get_models(),
        lambda: resources.get_methods("x.y"),
        lambda: resources.get_model_docs("x.y"),
    ],
    ids=["schema", "model", "models", "methods", "docs"],
)
def test_large_dynamic_resources_emit_compact_json(call):
    client = _schema_client()
    client.search_read.side_effect = _docs_client().search_read.side_effect
    with (
        patch.object(resources, "get_odoo_client", return_value=client),
        patch.object(resources, "_get_live_doc", return_value=None),
    ):
        output = call()
    json.loads(output)  # still valid JSON
    assert "\n" not in output, "pretty-printed output inflates token cost by ~25 %"


# ----- P3: odoo://model/{m} goes through the shared, validated fetch -----


def test_model_info_reads_fields_through_the_shared_cache():
    client = _schema_client()
    with patch.object(resources, "get_odoo_client", return_value=client):
        first = json.loads(resources.get_model_info("x.y"))
        second = json.loads(resources.get_model_info("x.y"))
    assert first["fields"] == _FIELDS and second["fields"] == _FIELDS
    assert client.get_model_fields.call_count == 1


def test_model_info_rejects_a_malformed_model_name_without_calling_odoo():
    client = _schema_client()
    with patch.object(resources, "get_odoo_client", return_value=client):
        payload = json.loads(resources.get_model_info("Not A Model"))
    assert "error" in payload and "Invalid model name" in payload["error"]
    assert client.get_model_fields.call_count == 0
    assert client.get_model_info.call_count == 0


def test_model_docs_selection_query_is_not_capped():
    """The per-field loop capped each field at 50 options; a single capped query would instead drop
    whole fields once the model's combined option count passed the cap. The query is bounded by the
    model (a few hundred rows at most), so it must not carry a limit."""
    client = _docs_client()
    with patch.object(resources, "get_odoo_client", return_value=client):
        resources.get_model_docs("x.y")
    (call,) = _selection_calls(client)
    assert "limit" not in call.kwargs
