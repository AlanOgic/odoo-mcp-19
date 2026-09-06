"""Method catalog behind ``odoo://methods/{model}``.

The static part of the catalog (descriptions and notes for the common ORM
methods) lives here; the *parameter names* are derived from
``arg_mapping.V2_ARG_MAPPING`` rather than restated, so the resource can never
again publish a name the JSON-2 client would reject (``default_get`` was
advertised as ``fields_list`` while the mapping — and Odoo — say ``fields``).
``tests/test_method_catalog.py`` pins the parity.

Live enrichment from ``/doc-bearer/`` is applied by a single helper to both
the static entries and the additional methods it discovers, so the two paths
cannot drift in what they expose.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .arg_mapping import V2_ARG_MAPPING
from .constants import MODULE_KNOWLEDGE
from .utils import _strip_html

# ORM categories carry params derived from V2_ARG_MAPPING; special_methods come from module_knowledge.json.
ORM_CATEGORIES: tuple[str, ...] = ("read_methods", "write_methods", "introspection_methods")
STATIC_CATEGORIES: tuple[str, ...] = ORM_CATEGORIES + ("special_methods",)

# (name, description, note) per category — params come from V2_ARG_MAPPING.
_STATIC_TABLE: Dict[str, List[tuple[str, str, Optional[str]]]] = {
    "read_methods": [
        ("search", "Search for record IDs", None),
        (
            "search_read",
            "Search and read in one call",
            "Extra read() options such as load=None go in kwargs_json.",
        ),
        (
            "read",
            "Read specific records by ID",
            "load='_classic_read' (default) returns Many2one as (id, name); load=None returns raw ID for better performance",
        ),
        ("search_count", "Count matching records", None),
        (
            "read_group",
            "Aggregation with grouping (deprecated in v19, use formatted_read_group)",
            "Deprecated in v19. Use formatted_read_group instead. Still works for backward compatibility.",
        ),
        (
            "formatted_read_group",
            "Aggregation with grouping (v19+ replacement for read_group)",
            "Uses 'aggregates' param with 'field:agg' format (e.g. 'amount_total:sum', '__count'). Replaces deprecated read_group.",
        ),
    ],
    "write_methods": [
        ("create", "Create new record(s)", "Pass list of dicts for batch creation"),
        ("write", "Update existing record(s)", None),
        ("unlink", "Delete record(s)", None),
        ("copy", "Duplicate record(s)", "Record-bound: the ids to duplicate go in 'ids', overrides in 'default'."),
    ],
    "introspection_methods": [
        ("fields_get", "Get field definitions and metadata", None),
        (
            "default_get",
            "Get default values for fields",
            "The parameter is 'fields' (the Odoo docstring's 'fields_list' is stale).",
        ),
        ("name_search", "Search by name (autocomplete)", None),
        (
            "check_access_rights",
            "Check user permissions (legacy, still works)",
            "Still works but has_access is preferred in v19+.",
        ),
        (
            "has_access",
            "Check if user has access (returns boolean)",
            "Preferred over check_access_rights in v19+. Returns True/False without raising exceptions.",
        ),
    ],
}

# Live-doc keys copied onto a catalog entry, in the shape the resource publishes.
_LIVE_SCALAR_KEYS: tuple[tuple[str, str], ...] = (("signature", "signature"), ("api", "api"), ("module", "module"))


def _static_entry(name: str, description: str, note: Optional[str]) -> Dict[str, Any]:
    """One catalog row; ``params`` is the mapping's positional order."""
    params = [param for _, param in sorted(V2_ARG_MAPPING[name])]
    entry: Dict[str, Any] = {"name": name, "description": description, "params": params}
    if note:
        entry["note"] = note
    return entry


def _special_methods(model_name: str) -> tuple[List[Dict[str, Any]], List[str], Optional[Dict[str, Any]]]:
    """Special methods, warnings and field mappings from module_knowledge.json."""
    special: List[Dict[str, Any]] = []
    warnings: List[str] = []
    field_mappings: Optional[Dict[str, Any]] = None
    for module_info in MODULE_KNOWLEDGE.get("modules", {}).values():
        if module_info.get("model") != model_name:
            continue
        for method_name, method_info in module_info.get("special_methods", {}).items():
            special.append(
                {
                    "name": method_name,
                    "description": method_info.get("description", ""),
                    "params": method_info.get("params", {}),
                    "instead_of": method_info.get("instead_of"),
                    "requires_ids": method_info.get("requires_ids", False),
                }
            )
        if module_info.get("notes"):
            warnings.append(module_info["notes"])
        if module_info.get("field_mappings"):
            field_mappings = module_info["field_mappings"]
    return special, warnings, field_mappings


def _param_details(parameters: Dict[str, Any]) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    for pname, pinfo in parameters.items():
        detail: Dict[str, Any] = {}
        if pinfo.get("annotation"):
            detail["type"] = pinfo["annotation"]
        if "default" in pinfo:
            detail["default"] = pinfo["default"]
        if pinfo.get("doc"):
            detail["description"] = _strip_html(pinfo["doc"])
        if detail:
            details[pname] = detail
    return details


def _enrich_from_live(entry: Dict[str, Any], live: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``entry`` extended with the live /doc-bearer/ facts for that method."""
    enriched = dict(entry)
    for live_key, out_key in _LIVE_SCALAR_KEYS:
        if live.get(live_key):
            enriched[out_key] = live[live_key]
    if live.get("return", {}).get("annotation"):
        enriched["return_type"] = live["return"]["annotation"]
    if live.get("raise"):
        enriched["exceptions"] = {k: _strip_html(v) for k, v in live["raise"].items()}
    details = _param_details(live.get("parameters") or {})
    if details:
        enriched["param_details"] = details
    return enriched


def _discovered_methods(live_methods: Dict[str, Any], known: Iterable[str]) -> List[Dict[str, Any]]:
    """Model-specific methods present in the live doc but absent from the catalog."""
    known_names = set(known)
    additional = []
    for name, live in live_methods.items():
        if name in known_names:
            continue
        entry: Dict[str, Any] = {
            "name": name,
            "description": _strip_html(live.get("doc", "")) if live.get("doc") else "",
        }
        if live.get("parameters"):
            entry["params"] = list(live["parameters"].keys())
        additional.append(_enrich_from_live(entry, live))
    additional.sort(key=lambda m: (m.get("module", "zzz"), m["name"]))
    return additional


def build_methods_payload(model_name: str, live_doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the ``odoo://methods/{model}`` document (pure; no I/O)."""
    payload: Dict[str, Any] = {
        category: [_static_entry(*row) for row in rows] for category, rows in _STATIC_TABLE.items()
    }
    special, warnings, field_mappings = _special_methods(model_name)
    payload["special_methods"] = special
    payload["warnings"] = warnings
    if field_mappings:
        payload["field_mappings"] = field_mappings
    payload["note"] = f"Use execute_method tool to call these on {model_name}"

    if not live_doc:
        payload["_source"] = "static (live docs unavailable)"
        return payload

    live_methods = live_doc.get("methods", {})
    for category in STATIC_CATEGORIES:
        payload[category] = [
            _enrich_from_live(entry, live_methods[entry["name"]]) if entry["name"] in live_methods else entry
            for entry in payload[category]
        ]
    known = [entry["name"] for category in STATIC_CATEGORIES for entry in payload[category]]
    additional = _discovered_methods(live_methods, known)
    if additional:
        payload["additional_methods"] = additional
    payload["_source"] = "live (enriched from /doc-bearer/)"
    return payload
