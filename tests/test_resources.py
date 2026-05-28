"""Unit tests for odoo:// schema resource handlers — error handling & validation.

These tests pin the v1.14.x fixes for the resource layer:
  * malformed model names are rejected with a clean message (regex validation
    was previously only enforced on the tool path, not the resource path);
  * a non-existent model surfaces a meaningful "not found" error instead of the
    cryptic ``'str' object has no attribute 'get'`` that leaked when the
    ``get_model_fields`` error sentinel was fed into the schema builders;
  * ``odoo://bundle`` reports invalid/missing models in its ``errors`` map
    rather than silently dropping them.

No live Odoo is needed: ``get_odoo_client`` is patched with a stub.
"""

import json
from unittest.mock import MagicMock, patch

import odoo_mcp.resources as resources

# A minimal but realistic fields_get payload for a valid model.
_VALID_FIELDS = {
    "id": {"type": "integer", "string": "ID", "readonly": True},
    "name": {"type": "char", "string": "Name", "required": True},
    "partner_id": {"type": "many2one", "string": "Partner", "relation": "res.partner"},
    "state": {
        "type": "selection",
        "string": "Status",
        "selection": [["draft", "Draft"], ["done", "Done"]],
    },
}

# The error sentinel shape returned by OdooClient.get_model_fields on failure.
_ERROR_SENTINEL = {"error": "Model this.does.not.exist not found"}

_CRYPTIC = "object has no attribute"


def _stub_client(fields_by_model):
    """Return a MagicMock client whose get_model_fields looks up per-model."""
    client = MagicMock()
    client.get_model_fields.side_effect = lambda model: fields_by_model[model]
    return client


# ----- quick-schema -----


def test_quick_schema_valid_model_builds_compact_schema():
    with patch.object(resources, "get_odoo_client", return_value=_stub_client({"res.partner": _VALID_FIELDS})):
        out = json.loads(resources.get_model_quick_schema("res.partner"))
    assert out["model"] == "res.partner"
    assert out["field_count"] == 4
    assert out["fields"]["name"]["req"] is True
    assert out["fields"]["partner_id"]["rel"] == "res.partner"
    assert "error" not in out


def test_quick_schema_invalid_format_is_rejected_cleanly():
    # Malformed name must never reach the API or the schema builder.
    out = json.loads(resources.get_model_quick_schema("INVALID-MODEL!"))
    assert "error" in out
    assert "Invalid model name format" in out["error"]
    assert _CRYPTIC not in out["error"]


def test_quick_schema_nonexistent_model_surfaces_meaningful_error():
    with patch.object(
        resources, "get_odoo_client", return_value=_stub_client({"this.does.not.exist": _ERROR_SENTINEL})
    ):
        out = json.loads(resources.get_model_quick_schema("this.does.not.exist"))
    assert "error" in out
    assert _CRYPTIC not in out["error"]
    assert "this.does.not.exist" in out["error"]


# ----- fields (light) -----


def test_fields_light_nonexistent_model_surfaces_meaningful_error():
    with patch.object(
        resources, "get_odoo_client", return_value=_stub_client({"this.does.not.exist": _ERROR_SENTINEL})
    ):
        out = json.loads(resources.get_model_fields_light("this.does.not.exist"))
    assert "error" in out
    assert _CRYPTIC not in out["error"]


# ----- schema (full) -----


def test_schema_nonexistent_model_surfaces_meaningful_error():
    with patch.object(
        resources, "get_odoo_client", return_value=_stub_client({"this.does.not.exist": _ERROR_SENTINEL})
    ):
        out = json.loads(resources.get_model_schema("this.does.not.exist"))
    assert "error" in out
    assert _CRYPTIC not in out["error"]


# ----- bundle -----


def test_bundle_reports_invalid_model_in_errors_not_silently_dropped():
    fields_by_model = {
        "res.partner": _VALID_FIELDS,
        "this.does.not.exist": _ERROR_SENTINEL,
    }
    with patch.object(resources, "get_odoo_client", return_value=_stub_client(fields_by_model)):
        out = json.loads(resources.get_bundle("res.partner,this.does.not.exist"))
    assert "res.partner" in out["models"]
    # The bad model must be surfaced, not silently omitted.
    assert "this.does.not.exist" in out["errors"]
    assert out["total"] == 1


# ----- find-model (concept resolution) -----


def test_find_model_exact_alias_still_resolves():
    with patch.object(resources, "get_odoo_client", return_value=MagicMock()):
        out = json.loads(resources.find_model_resource("invoice"))
    assert out["best_match"] == "account.move"
    assert out["source"] == "alias"


def test_find_model_multiword_concept_tokenizes_to_known_aliases():
    # "customer invoice" is not an exact alias nor a model substring, but both
    # words are known aliases — the union must surface instead of no match.
    with patch.object(resources, "get_odoo_client", return_value=MagicMock()):
        out = json.loads(resources.find_model_resource("customer invoice"))
    models = {m["model"] for m in out["all_matches"]}
    assert "res.partner" in models
    assert "account.move" in models
    assert out["best_match"] is not None
    assert out["source"] == "alias-token"


def test_find_model_multiword_partial_match_still_resolves():
    # Only one token is a known alias; that partial match beats no match.
    with patch.object(resources, "get_odoo_client", return_value=MagicMock()):
        out = json.loads(resources.find_model_resource("customer zzzznotareal"))
    models = {m["model"] for m in out["all_matches"]}
    assert "res.partner" in models
    assert out["source"] == "alias-token"
