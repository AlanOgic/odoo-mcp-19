"""Unit tests for the v1.18 API reference resources.

Three static references (``odoo://api/json2-protocol``, ``odoo://api/version-drift``,
``odoo://api/x2many-commands``) transcribed from the Odoo 19 source / docs, and one
live catalogue (``odoo://api-index``) built from ``/doc-bearer/index.json``.
The facts pinned here are the ones an LLM trained on Odoo <= 16 gets wrong.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

import odoo_mcp.app  # noqa: F401  (binds the mcp decorator first)
import odoo_mcp.resources as resources
import odoo_mcp.utils as utils
from odoo_mcp import api_reference
from odoo_mcp.app import mcp
from odoo_mcp.server import _RESOURCE_ROUTES, read_resource

# ----- odoo://api/json2-protocol -----


def test_json2_protocol_states_named_args_only_and_the_ids_rule():
    ref = api_reference.json2_protocol_reference()
    assert ref["endpoint"] == "POST {ODOO_URL}/json/2/{model}/{method}"
    assert set(ref["body"]) >= {"ids", "context"}
    assert ref["positional_args"] is False
    assert "@api.model" in ref["body"]["ids"] and "422" in ref["body"]["ids"]


def test_json2_protocol_error_shape_and_status_table():
    ref = api_reference.json2_protocol_reference()
    assert ref["error_body_keys"] == ["name", "message", "arguments", "context", "debug"]
    statuses = ref["status_codes"]
    assert "signature" in statuses["422"].lower()
    assert "private" in statuses["403"].lower() and "private" not in statuses["404"].lower()
    assert "403" in ref["private_methods"]
    assert "html" in statuses["415"].lower()
    assert "500" in statuses


def test_json2_protocol_documents_transaction_isolation_and_rate_limits():
    ref = api_reference.json2_protocol_reference()
    assert "own SQL transaction" in ref["transactions"]["rule"]
    assert "action_" in ref["transactions"]["advice"]
    assert ref["rate_limiting"]["odoo_core"] is None
    assert "429" in ref["rate_limiting"]["this_server"]


# ----- odoo://api/version-drift -----


@pytest.mark.parametrize(
    "old,new",
    [
        ("name_get", "display_name"),
        ("args", "domain"),
        ("read_group", "formatted_read_group"),
        ("check_access_rights", "check_access"),
        ("group_operator", "aggregator"),
    ],
)
def test_version_drift_maps_each_legacy_name_to_its_replacement(old, new):
    entries = {e["old"]: e for e in api_reference.version_drift_reference()["changes"]}
    assert new in entries[old]["use"]
    assert entries[old]["since"]
    assert entries[old]["pr"].startswith("https://github.com/odoo/odoo/pull/")


def test_version_drift_never_claims_a_deprecated_method_is_gone():
    """check_access_rights / check_access_rule are @api.deprecated in 19.0, not removed nor private."""
    entries = {e["old"]: e for e in api_reference.version_drift_reference()["changes"]}
    for old in ("check_access_rights", "check_access_rule"):
        assert "deprecated" in entries[old]["json2_impact"].lower()
        assert "404" not in entries[old]["json2_impact"]


def test_version_drift_reports_private_methods_as_403():
    changes = api_reference.version_drift_reference()["changes"]
    private_entries = [e for e in changes if "@api.private" in e["use"] or "search_fetch" in e["use"]]
    assert private_entries
    assert all("403" in e["json2_impact"] for e in private_entries)


def test_version_drift_entries_carry_the_json2_impact():
    changes = api_reference.version_drift_reference()["changes"]
    assert all({"since", "old", "use", "json2_impact", "pr"} <= set(e) for e in changes)
    assert any("@api.private" in e["old"] or "@api.private" in e["use"] for e in changes)


# ----- odoo://api/x2many-commands -----


def test_x2many_commands_lists_the_seven_literal_triples():
    ref = api_reference.x2many_commands_reference()
    codes = {c["code"]: c for c in ref["commands"]}
    assert sorted(codes) == [0, 1, 2, 3, 4, 5, 6]
    assert codes[0]["shape"] == "[0, 0, values]"
    assert codes[6]["shape"] == "[6, 0, ids]"
    assert "literal" in ref["rpc_rule"].lower()
    assert "many2many" in codes[0]["notes"].lower()
    assert "cascade" in codes[3]["notes"].lower()


# ----- odoo://api-index -----

_RAW_INDEX = {
    "modules": ["base", "web", "sale"],
    "models": [
        {
            "model": "sale.order",
            "name": "Sales Order",
            "fields": {"name": {"string": "Order Reference"}, "state": {"string": "Status"}},
            "methods": ["action_confirm", "read"],
        },
        {"model": "res.partner", "name": "Contact", "fields": {"name": {"string": "Name"}}, "methods": ["read"]},
    ],
}


def test_build_api_index_keeps_names_and_counts_but_drops_field_maps():
    out = api_reference.build_api_index(_RAW_INDEX)
    assert out["module_count"] == 3 and out["model_count"] == 2
    assert out["modules"] == ["base", "web", "sale"]
    assert out["models"] == [
        {"model": "res.partner", "name": "Contact", "field_count": 1, "method_count": 1},
        {"model": "sale.order", "name": "Sales Order", "field_count": 2, "method_count": 2},
    ]


@pytest.fixture(autouse=True)
def _clear_api_index_cache():
    utils.clear_api_index_cache()
    yield
    utils.clear_api_index_cache()


def _client(index, url="https://a.example.com", username="bot"):
    client = MagicMock()
    client.url, client.username = url, username
    client.get_api_index.return_value = index
    return client


def test_api_index_resource_returns_the_compact_catalogue():
    with patch.object(utils, "get_odoo_client", return_value=_client(_RAW_INDEX)):
        out = json.loads(resources.get_api_index())
    assert out["model_count"] == 2
    assert out["source"] == "/doc-bearer/index.json"


def test_api_index_unavailable_doc_endpoint_gives_an_actionable_error():
    with patch.object(utils, "get_odoo_client", return_value=_client(None)):
        out = json.loads(resources.get_api_index())
    assert "api_doc.group_allow_doc" in out["hint"]
    assert "odoo://models" in out["hint"]


def test_api_index_cache_is_bounded():
    from odoo_mcp.constants import _API_INDEX_CACHE, _API_INDEX_CACHE_MAX_ENTRIES

    for i in range(_API_INDEX_CACHE_MAX_ENTRIES + 3):
        with patch.object(utils, "get_odoo_client", return_value=_client(_RAW_INDEX, username=f"user{i}")):
            resources.get_api_index()
    assert len(_API_INDEX_CACHE) == _API_INDEX_CACHE_MAX_ENTRIES


def test_api_index_is_cached_per_client_identity():
    shared = _client(_RAW_INDEX)
    other = _client(_RAW_INDEX, username="someone-else")
    with patch.object(utils, "get_odoo_client", return_value=shared):
        resources.get_api_index()
        resources.get_api_index()
    with patch.object(utils, "get_odoo_client", return_value=other):
        resources.get_api_index()
    assert shared.get_api_index.call_count == 1
    assert other.get_api_index.call_count == 1


# ----- bridge routes + parity -----


@pytest.mark.parametrize("uri", ["odoo://api/json2-protocol", "odoo://api/version-drift", "odoo://api/x2many-commands"])
def test_bridge_routes_the_static_reference_uris(uri):
    out = json.loads(read_resource(uri, max_chars=0))
    assert "error" not in out


def test_bridge_routes_api_index():
    with patch.object(utils, "get_odoo_client", return_value=_client(_RAW_INDEX)):
        out = json.loads(read_resource("odoo://api-index", max_chars=0))
    assert out["model_count"] == 2


def test_every_registered_resource_has_exactly_one_bridge_route():
    """Adding a resource without a route silently 404s read_resource; this pins parity."""
    statics = asyncio.run(mcp.list_resources(run_middleware=False))
    templates = asyncio.run(mcp.list_resource_templates(run_middleware=False))
    uris = [str(r.uri) for r in statics] + [t.uri_template for t in templates]
    samples = {
        "{model_name}": "res.partner",
        "{record_id}": "1",
        "{models_csv}": "res.partner,sale.order",
        "{concept}": "invoice",
        "{query}": "invoice",
        "{model}": "sale.order",
        "{target}": "sale",
        "{module_name}": "sale",
    }
    for uri in uris:
        for placeholder, value in samples.items():
            uri = uri.replace(placeholder, value)
        assert "{" not in uri, f"unmapped placeholder in {uri}"
        matches = [p.pattern for p, _, _ in _RESOURCE_ROUTES if p.match(uri)]
        assert len(matches) == 1, f"expected exactly one bridge route for {uri}, got {matches}"
    assert len(uris) == len(_RESOURCE_ROUTES)
