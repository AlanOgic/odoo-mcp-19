"""
MCP Resource handlers for the Odoo MCP Server.

All 27 @mcp.resource decorated functions. Importing this module
registers all resources with the FastMCP instance.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from .app import mcp
from .constants import (
    CONCEPT_ALIASES,
    ERROR_CATEGORIES,
    MODEL_STATE_MACHINES,
    MODULE_KNOWLEDGE,
    RUNTIME_MODEL_ISSUES,
    TOOL_REGISTRY,
    _DEFAULT_BOOTSTRAP_MODELS,
    _RUNTIME_ISSUES_LOCK,
    _validate_model,
)
from .odoo_client import get_odoo_client
from .utils import (
    _build_compact_schema,
    _get_documentation_urls,
    _get_live_doc,
    _get_module_knowledge,
    _get_module_knowledge_by_name,
    _strip_html,
)


# ----- Internal helpers -----

# Hint appended to model-resolution errors so agents know where to look next.
_MODEL_LOOKUP_HINT = "Use odoo://models or odoo://find-model/{concept} to find the right model."


def _fetch_model_fields(model_name: str) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    """Validate a model name and fetch its ``fields_get`` definition.

    Returns ``(fields, None)`` on success or ``(None, error_dict)`` when the
    name is malformed or the model does not exist / is inaccessible.

    Centralizing this guard keeps the schema builders from ever receiving the
    ``{"error": ...}`` sentinel that ``OdooClient.get_model_fields`` returns on
    failure — feeding that sentinel into the builders previously leaked the
    cryptic ``'str' object has no attribute 'get'`` (the error string was
    iterated as if it were field metadata). The regex check also mirrors the
    validation enforced on the tool path, which the resource path lacked.
    """
    err = _validate_model(model_name)
    if err:
        return None, {"error": err, "hint": _MODEL_LOOKUP_HINT}
    fields = get_odoo_client().get_model_fields(model_name)
    # get_model_fields returns {"error": "<msg>"} on failure. A real field can
    # be named "error", but its value is always a dict, never a str — so an
    # str-valued "error" key unambiguously identifies the failure sentinel.
    if isinstance(fields, dict) and isinstance(fields.get("error"), str):
        return None, {
            "error": f"Model '{model_name}' not found or inaccessible: {fields['error']}",
            "hint": _MODEL_LOOKUP_HINT,
        }
    return fields, None


# ----- MCP Resources -----


@mcp.resource(
    "odoo://models",
    description="List all available models in Odoo",
)
def get_models() -> str:
    """Lists all available models"""
    odoo_client = get_odoo_client()
    models = odoo_client.get_models()
    return json.dumps(models, indent=2)


@mcp.resource(
    "odoo://model/{model_name}",
    description="Get information about a specific model including fields",
)
def get_model_info(model_name: str) -> str:
    """Get information about a specific model"""
    odoo_client = get_odoo_client()
    try:
        model_info = odoo_client.get_model_info(model_name)
        fields = odoo_client.get_model_fields(model_name)
        model_info["fields"] = fields
        return json.dumps(model_info, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource(
    "odoo://model/{model_name}/schema",
    description="Complete schema for a model including fields and relationships",
)
def get_model_schema(model_name: str) -> str:
    """Get comprehensive schema information for a model"""
    fields, error = _fetch_model_fields(model_name)
    if fields is None:
        return json.dumps(error, indent=2)
    try:
        schema = {
            "model": model_name,
            "fields": fields,
            "relationships": {},
            "required_fields": [],
            "selection_fields": {},
        }

        for field_name, field_def in fields.items():
            field_type = field_def.get('type', '')

            if field_type in ['many2one', 'one2many', 'many2many']:
                schema['relationships'][field_name] = {
                    'type': field_type,
                    'relation': field_def.get('relation', ''),
                    'string': field_def.get('string', '')
                }

            if field_def.get('required'):
                schema['required_fields'].append(field_name)

            if field_type == 'selection' and field_def.get('selection'):
                schema['selection_fields'][field_name] = {
                    'label': field_def.get('string'),
                    'values': field_def.get('selection'),
                }

        return json.dumps(schema, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource(
    "odoo://model/{model_name}/fields",
    description="Lightweight field list: names, types, labels (much smaller than /schema)",
)
def get_model_fields_light(model_name: str) -> str:
    """Get a compact field summary for a model.

    Returns only essential info per field: type, label, required flag,
    relation model (for relational fields), and selection values.
    Much smaller than /schema; use /quick-schema for the densest form.
    """
    fields, error = _fetch_model_fields(model_name)
    if fields is None:
        return json.dumps(error, indent=2)
    try:
        light = {}
        for name, meta in fields.items():
            entry = {
                "type": meta.get("type", ""),
                "string": meta.get("string", ""),
                "required": meta.get("required", False),
            }
            if meta.get("type") in ("many2one", "one2many", "many2many"):
                entry["relation"] = meta.get("relation", "")
            if meta.get("type") == "selection" and meta.get("selection"):
                entry["selection"] = meta.get("selection")
            light[name] = entry
        return json.dumps({"model": model_name, "field_count": len(light), "fields": light}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource(
    "odoo://model/{model_name}/quick-schema",
    description="Ultra-compact schema (~1.5KB): short keys, no labels, no help. Best for token savings.",
)
def get_model_quick_schema(model_name: str) -> str:
    """Get ultra-compact schema for a model.

    Returns minimal field info with short keys (t=type, req=required, ro=readonly, rel=relation).
    No indentation, no labels, no help text. Typically 60-80% smaller than /fields.
    """
    fields, error = _fetch_model_fields(model_name)
    if fields is None:
        return json.dumps(error)
    try:
        schema = _build_compact_schema(fields)
        schema["model"] = model_name
        schema["field_count"] = len(schema["fields"])
        return json.dumps(schema, separators=(",", ":"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource(
    "odoo://model/{model_name}/workflow",
    description="State machine transitions for a model: states, methods, side effects, irreversibility",
)
def get_model_workflow(model_name: str) -> str:
    """Get workflow/state machine for a model.

    Returns state transitions with methods to call, side effects, and irreversibility flags.
    Static data for 6 main models, dynamic fallback for others (reads state field + action methods).
    """
    # Static state machines for well-known models
    if model_name in MODEL_STATE_MACHINES:
        result = {
            "model": model_name,
            "source": "static",
            **MODEL_STATE_MACHINES[model_name],
        }
        return json.dumps(result, indent=2)

    # Dynamic fallback: try to infer workflow from model metadata
    odoo_client = get_odoo_client()
    try:
        fields = odoo_client.get_model_fields(model_name)
        result = {
            "model": model_name,
            "source": "dynamic",
            "state_field": None,
            "states": [],
            "available_methods": [],
        }

        # Check for state field
        if "state" in fields:
            state_meta = fields["state"]
            if state_meta.get("type") == "selection" and state_meta.get("selection"):
                result["state_field"] = "state"
                result["states"] = [
                    {"value": val, "label": label}
                    for val, label in state_meta["selection"]
                ]

        # Check live doc for action/button methods
        live_doc = _get_live_doc(model_name)
        if live_doc:
            for method_name, method_info in live_doc.get("methods", {}).items():
                if method_name.startswith(("action_", "button_")):
                    entry = {
                        "name": method_name,
                        "description": _strip_html(method_info.get("doc", "")) if method_info.get("doc") else "",
                    }
                    if method_info.get("api"):
                        entry["api"] = method_info["api"]
                    result["available_methods"].append(entry)

        # Also check module knowledge
        for module_name, module_info in MODULE_KNOWLEDGE.get("modules", {}).items():
            if module_info.get("model") == model_name:
                for method_name, method_info in module_info.get("special_methods", {}).items():
                    if method_name.startswith(("action_", "button_")):
                        result["available_methods"].append({
                            "name": method_name,
                            "description": method_info.get("description", ""),
                        })

        if not result["state_field"] and not result["available_methods"]:
            result["note"] = f"No state machine detected for {model_name}. It may not have workflow transitions."

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource(
    "odoo://bundle/{models_csv}",
    description="Batch quick-schema for multiple models in one call (comma-separated, max 10)",
)
def get_bundle(models_csv: str) -> str:
    """Get compact schemas for multiple models in a single call.

    URI format: odoo://bundle/res.partner,sale.order,stock.picking
    Max 10 models per request to keep response size reasonable.

    Schema fetches run in parallel (one thread per model, capped at 10) so
    bundle latency is bounded by the slowest fetch, not their sum.
    """
    model_names = [m.strip() for m in models_csv.split(",") if m.strip()]
    if len(model_names) > 10:
        return json.dumps({"error": "Maximum 10 models per bundle request", "requested": len(model_names)})

    bundle: Dict[str, Any] = {"models": {}, "errors": {}}

    def _fetch(name: str) -> tuple[str, Any]:
        fields, error = _fetch_model_fields(name)
        if fields is None:
            return name, error
        try:
            schema = _build_compact_schema(fields)
            schema["field_count"] = len(schema["fields"])
            return name, schema
        except Exception as exc:
            return name, {"error": str(exc)}

    if model_names:
        with ThreadPoolExecutor(max_workers=min(len(model_names), 10)) as executor:
            for name, payload in executor.map(_fetch, model_names):
                if isinstance(payload, dict) and "error" in payload and "fields" not in payload:
                    bundle["errors"][name] = payload["error"]
                else:
                    bundle["models"][name] = payload

    bundle["total"] = len(bundle["models"])
    return json.dumps(bundle, separators=(",", ":"))


@mcp.resource(
    "odoo://session-bootstrap",
    description="Bootstrap a conversation: quick-schemas + workflows for common models (configurable via MCP_BOOTSTRAP_MODELS)",
)
def get_session_bootstrap() -> str:
    """Single call to bootstrap a conversation with schemas and workflows.

    Includes compact schemas and workflow state machines for the most common models.
    Configure via MCP_BOOTSTRAP_MODELS env var (comma-separated model names).
    Default: res.partner,sale.order,account.move,product.product,stock.picking
    """
    odoo_client = get_odoo_client()
    models_csv = os.environ.get("MCP_BOOTSTRAP_MODELS", _DEFAULT_BOOTSTRAP_MODELS)
    model_names = [m.strip() for m in models_csv.split(",") if m.strip()][:20]

    result: Dict[str, Any] = {"schemas": {}, "workflows": {}, "errors": {}}

    def _fetch(name: str) -> tuple[str, Any]:
        try:
            fields = odoo_client.get_model_fields(name)
            schema = _build_compact_schema(fields)
            schema["field_count"] = len(schema["fields"])
            return name, schema
        except Exception as exc:
            return name, exc

    # Schema fetches run in parallel (capped at 10 workers, well within the
    # OdooClient session's 20-conn pool so a concurrent execute_method still
    # has headroom). Workflows are read from the in-memory state-machine table.
    if model_names:
        with ThreadPoolExecutor(max_workers=min(len(model_names), 10)) as executor:
            for name, payload in executor.map(_fetch, model_names):
                if isinstance(payload, Exception):
                    result["errors"][name] = str(payload)
                else:
                    result["schemas"][name] = payload

    for model_name in model_names:
        if model_name in MODEL_STATE_MACHINES:
            result["workflows"][model_name] = MODEL_STATE_MACHINES[model_name]

    result["models_loaded"] = len(result["schemas"])
    result["workflows_loaded"] = len(result["workflows"])
    return json.dumps(result, separators=(",", ":"))


@mcp.resource(
    "odoo://record/{model_name}/{record_id}",
    description="Get a specific record by ID",
)
def get_record(model_name: str, record_id: str) -> str:
    """Get a specific record by ID"""
    odoo_client = get_odoo_client()
    try:
        if not record_id or record_id == 'None':
            return json.dumps({"error": "Record ID is required"}, indent=2)

        record = odoo_client.read_records(model_name, [int(record_id)])
        if not record:
            return json.dumps({"error": f"Record not found: {model_name} ID {record_id}"}, indent=2)
        return json.dumps(record[0], indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource(
    "odoo://methods/{model_name}",
    description="Available methods for a model including module-specific special methods",
)
def get_methods(model_name: str) -> str:
    """Get available methods for a model, including special methods from knowledge base"""
    common_methods = {
        "read_methods": [
            {"name": "search", "description": "Search for record IDs", "params": ["domain", "offset", "limit", "order"]},
            {"name": "search_read", "description": "Search and read in one call", "params": ["domain", "fields", "offset", "limit", "order", "load"]},
            {"name": "read", "description": "Read specific records by ID", "params": ["ids", "fields", "load"], "note": "load='_classic_read' (default) returns Many2one as (id, name); load=None returns raw ID for better performance"},
            {"name": "search_count", "description": "Count matching records", "params": ["domain"]},
            {"name": "read_group", "description": "Aggregation with grouping (deprecated in v19, use formatted_read_group)", "params": ["domain", "fields", "groupby", "offset", "limit", "orderby", "lazy"], "note": "Deprecated in v19. Use formatted_read_group instead. Still works for backward compatibility."},
            {"name": "formatted_read_group", "description": "Aggregation with grouping (v19+ replacement for read_group)", "params": ["domain", "groupby", "aggregates", "having", "offset", "limit", "order"], "note": "Uses 'aggregates' param with 'field:agg' format (e.g. 'amount_total:sum', '__count'). Replaces deprecated read_group."},
        ],
        "write_methods": [
            {"name": "create", "description": "Create new record(s)", "params": ["vals_list"], "note": "Pass list of dicts for batch creation"},
            {"name": "write", "description": "Update existing record(s)", "params": ["ids", "vals"]},
            {"name": "unlink", "description": "Delete record(s)", "params": ["ids"]},
            {"name": "copy", "description": "Duplicate a record", "params": ["id", "default"]},
        ],
        "introspection_methods": [
            {"name": "fields_get", "description": "Get field definitions and metadata", "params": ["allfields", "attributes"]},
            {"name": "default_get", "description": "Get default values for fields", "params": ["fields_list"]},
            {"name": "name_search", "description": "Search by name (autocomplete)", "params": ["name", "domain", "operator", "limit"]},
            {"name": "check_access_rights", "description": "Check user permissions (legacy, still works)", "params": ["operation", "raise_exception"], "note": "Still works but has_access is preferred in v19+."},
            {"name": "has_access", "description": "Check if user has access (returns boolean)", "params": ["operation"], "note": "Preferred over check_access_rights in v19+. Returns True/False without raising exceptions."},
        ],
        "special_methods": [],
        "warnings": [],
        "note": f"Use execute_method tool to call these on {model_name}",
    }

    # Check module knowledge for special methods
    for module_name, module_info in MODULE_KNOWLEDGE.get("modules", {}).items():
        if module_info.get("model") == model_name:
            special_methods = module_info.get("special_methods", {})
            for method_name, method_info in special_methods.items():
                common_methods["special_methods"].append({
                    "name": method_name,
                    "description": method_info.get("description", ""),
                    "params": method_info.get("params", {}),
                    "instead_of": method_info.get("instead_of"),
                    "requires_ids": method_info.get("requires_ids", False)
                })

            # Add warnings if any
            if module_info.get("notes"):
                common_methods["warnings"].append(module_info["notes"])

            # Add field mappings if any
            if module_info.get("field_mappings"):
                common_methods["field_mappings"] = module_info["field_mappings"]

    # --- Live enrichment from /doc-bearer/ ---
    live_doc = _get_live_doc(model_name)
    if live_doc:
        live_methods = live_doc.get("methods", {})

        # Build set of all static method names for quick lookup
        static_names = set()
        for category in ["read_methods", "write_methods", "introspection_methods", "special_methods"]:
            for m in common_methods.get(category, []):
                static_names.add(m["name"])

        # 1) Enrich existing static methods with live signatures, types, decorators
        for category in ["read_methods", "write_methods", "introspection_methods", "special_methods"]:
            for method_entry in common_methods.get(category, []):
                name = method_entry["name"]
                if name in live_methods:
                    live = live_methods[name]
                    if live.get("signature"):
                        method_entry["signature"] = live["signature"]
                    if live.get("return"):
                        ret = live["return"]
                        if ret.get("annotation"):
                            method_entry["return_type"] = ret["annotation"]
                    if live.get("api"):
                        method_entry["api"] = live["api"]
                    if live.get("module"):
                        method_entry["module"] = live["module"]
                    if live.get("raise"):
                        method_entry["exceptions"] = {
                            k: _strip_html(v) for k, v in live["raise"].items()
                        }
                    # Enrich params with types and defaults from live data
                    if live.get("parameters"):
                        param_details = {}
                        for pname, pinfo in live["parameters"].items():
                            detail = {}
                            if pinfo.get("annotation"):
                                detail["type"] = pinfo["annotation"]
                            if "default" in pinfo:
                                detail["default"] = pinfo["default"]
                            if pinfo.get("doc"):
                                detail["description"] = _strip_html(pinfo["doc"])
                            if detail:
                                param_details[pname] = detail
                        if param_details:
                            method_entry["param_details"] = param_details

        # 2) Discover additional model-specific methods not in our static list
        additional = []
        for name, live in live_methods.items():
            if name not in static_names:
                entry = {
                    "name": name,
                    "description": _strip_html(live.get("doc", "")) if live.get("doc") else "",
                }
                if live.get("signature"):
                    entry["signature"] = live["signature"]
                if live.get("return", {}).get("annotation"):
                    entry["return_type"] = live["return"]["annotation"]
                if live.get("api"):
                    entry["api"] = live["api"]
                if live.get("module"):
                    entry["module"] = live["module"]
                if live.get("raise"):
                    entry["exceptions"] = {
                        k: _strip_html(v) for k, v in live["raise"].items()
                    }
                if live.get("parameters"):
                    entry["params"] = list(live["parameters"].keys())
                additional.append(entry)

        if additional:
            # Sort by module then name for consistent ordering
            additional.sort(key=lambda m: (m.get("module", "zzz"), m["name"]))
            common_methods["additional_methods"] = additional

        common_methods["_source"] = "live (enriched from /doc-bearer/)"
    else:
        common_methods["_source"] = "static (live docs unavailable)"

    return json.dumps(common_methods, indent=2)


@mcp.resource(
    "odoo://model/{model_name}/docs",
    description="Rich documentation for a model: labels, field help, selection options, action help",
)
def get_model_docs(model_name: str) -> str:
    """Get comprehensive documentation for a model from Odoo's internal metadata."""
    odoo_client = get_odoo_client()

    try:
        result = {
            "model": model_name,
            "model_info": {},
            "fields": {},
            "selection_options": {},
            "actions_help": [],
        }

        # 1. Get model info (display name, description)
        model_info = odoo_client.search_read(
            "ir.model",
            [["model", "=", model_name]],
            fields=["name", "info", "modules"],
            limit=1
        )
        if model_info:
            result["model_info"] = {
                "display_name": model_info[0].get("name"),
                "description": model_info[0].get("info"),
                "modules": model_info[0].get("modules"),
            }

        # 2. Get fields with labels and help text
        fields_meta = odoo_client.search_read(
            "ir.model.fields",
            [["model", "=", model_name]],
            fields=["name", "field_description", "help", "ttype", "relation", "required"],
            limit=200
        )
        for f in fields_meta:
            result["fields"][f["name"]] = {
                "label": f.get("field_description"),
                "type": f.get("ttype"),
                "help": f.get("help") or None,
                "relation": f.get("relation") or None,
                "required": f.get("required"),
            }

        # 3. Get selection field options with labels
        selection_fields = [f["name"] for f in fields_meta if f.get("ttype") == "selection"]
        if selection_fields:
            for field_name in selection_fields[:20]:  # Limit to avoid too many queries
                selections = odoo_client.search_read(
                    "ir.model.fields.selection",
                    [["field_id.model", "=", model_name], ["field_id.name", "=", field_name]],
                    fields=["value", "name", "sequence"],
                    limit=50
                )
                if selections:
                    result["selection_options"][field_name] = [
                        {"value": s["value"], "label": s["name"]}
                        for s in sorted(selections, key=lambda x: x.get("sequence", 0))
                    ]

        # 4. Get action help text (contextual documentation)
        actions = odoo_client.search_read(
            "ir.actions.act_window",
            [["res_model", "=", model_name]],
            fields=["name", "help"],
            limit=10
        )
        for action in actions:
            if action.get("help"):
                # Strip HTML tags for cleaner output
                help_text = re.sub(r'<[^>]+>', ' ', action.get("help", ""))
                help_text = ' '.join(help_text.split())  # Normalize whitespace
                result["actions_help"].append({
                    "action_name": action.get("name"),
                    "help": help_text
                })

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource(
    "odoo://concepts",
    description="Mapping of business concepts to Odoo model names (contact->res.partner, invoice->account.move)",
)
def get_concept_mappings() -> str:
    """Get all known concept-to-model mappings for natural language model discovery."""
    return json.dumps({
        "description": "Business concept to Odoo model mappings",
        "usage": "Use odoo://find-model/{concept} resource to search, or look up concepts here",
        "mappings": CONCEPT_ALIASES
    }, indent=2)


@mcp.resource(
    "odoo://templates",
    description="List all available resource templates with their URI patterns (for clients that don't support resources/templates/list)",
)
def get_resource_templates() -> str:
    """List all resource templates available in this MCP server.

    This is a workaround for MCP clients that only call resources/list
    and don't support resources/templates/list from the MCP spec.
    """
    templates = {
        "description": "Available resource templates (parameterized URIs)",
        "note": "Replace {param} with actual values when reading these resources",
        "templates": {
            "odoo://model/{model_name}/quick-schema": {
                "description": "Ultra-compact schema (~1.5KB): short keys (t=type, req, ro, rel). Best for token savings.",
                "example": "odoo://model/res.partner/quick-schema"
            },
            "odoo://model/{model_name}/workflow": {
                "description": "State machine: transitions, methods, side effects, irreversibility",
                "example": "odoo://model/sale.order/workflow"
            },
            "odoo://bundle/{models_csv}": {
                "description": "Batch quick-schema for multiple models (comma-separated, max 10)",
                "example": "odoo://bundle/res.partner,sale.order,stock.picking"
            },
            "odoo://session-bootstrap": {
                "description": "Bootstrap a conversation: quick-schemas + workflows for common models",
                "example": "odoo://session-bootstrap"
            },
            "odoo://model/{model_name}": {
                "description": "Get information about a specific model including fields",
                "example": "odoo://model/res.partner"
            },
            "odoo://model/{model_name}/schema": {
                "description": "Complete schema for a model including fields, relationships, required fields",
                "example": "odoo://model/sale.order/schema"
            },
            "odoo://model/{model_name}/fields": {
                "description": "Lightweight field list: names, types, labels, required flag (much smaller than /schema)",
                "example": "odoo://model/res.partner/fields"
            },
            "odoo://model/{model_name}/docs": {
                "description": "Rich documentation: labels, field help, selection options, action help",
                "example": "odoo://model/account.move/docs"
            },
            "odoo://methods/{model_name}": {
                "description": "Available methods for a model including special methods from knowledge base",
                "example": "odoo://methods/crm.lead"
            },
            "odoo://record/{model_name}/{record_id}": {
                "description": "Get a specific record by ID",
                "example": "odoo://record/res.partner/1"
            },
            "odoo://find-model/{concept}": {
                "description": "Find Odoo model from natural language concept (customer, invoice, etc.)",
                "example": "odoo://find-model/customer"
            },
            "odoo://actions/{model}": {
                "description": "Discover all available actions, workflows, and methods for a model",
                "example": "odoo://actions/sale.order"
            },
            "odoo://tools/{query}": {
                "description": "Search available operations by keyword",
                "example": "odoo://tools/invoice"
            },
            "odoo://docs/{target}": {
                "description": "Documentation URLs and GitHub links for any model or module",
                "example": "odoo://docs/sale"
            },
            "odoo://module-knowledge/{module_name}": {
                "description": "Get knowledge for a specific module (special methods, patterns)",
                "example": "odoo://module-knowledge/knowledge"
            },
        },
        "usage": "Read any template by replacing {param} with your value using ReadMcpResourceTool"
    }
    return json.dumps(templates, indent=2)


@mcp.resource(
    "odoo://module-knowledge",
    description="Module-specific methods and patterns knowledge base",
)
def get_module_knowledge() -> str:
    """Get the full module knowledge base with special methods and error patterns"""
    return _get_module_knowledge()


@mcp.resource(
    "odoo://module-knowledge/{module_name}",
    description="Get knowledge for a specific module (sale, crm, account, etc.)",
)
def get_module_knowledge_by_name(module_name: str) -> str:
    """Get specific module knowledge"""
    return _get_module_knowledge_by_name(module_name)


@mcp.resource(
    "odoo://docs/{target}",
    description="Documentation URLs and GitHub links for any Odoo model or module",
)
def get_documentation_urls(target: str) -> str:
    """Get documentation URLs for a model or module."""
    return _get_documentation_urls(target)


@mcp.resource(
    "odoo://workflows",
    description="Available business workflows based on installed modules",
)
def get_workflows() -> str:
    """Discover available business workflows"""
    odoo_client = get_odoo_client()
    try:
        modules = odoo_client.search_read(
            'ir.module.module',
            [('state', '=', 'installed')],
            fields=['name', 'shortdesc', 'application'],
            limit=500
        )

        module_names = {m['name']: m.get('shortdesc', '') for m in modules}
        workflows = {}

        if 'sale' in module_names:
            workflows['sales'] = {
                "module": "sale",
                "title": "Sales Management",
                "workflows": [{
                    "name": "quotation_to_order",
                    "steps": [
                        "Create quotation (sale.order with state='draft')",
                        "Confirm order (method: action_confirm)",
                        "Create invoice (method: _create_invoices)"
                    ],
                    "model": "sale.order"
                }]
            }

        if 'crm' in module_names:
            workflows['crm'] = {
                "module": "crm",
                "title": "CRM / Leads",
                "workflows": [{
                    "name": "lead_to_opportunity",
                    "steps": [
                        "Create lead (crm.lead)",
                        "Convert to opportunity",
                        "Mark as won (method: action_set_won)"
                    ],
                    "model": "crm.lead"
                }]
            }

        if 'account' in module_names:
            workflows['accounting'] = {
                "module": "account",
                "title": "Accounting",
                "workflows": [{
                    "name": "create_invoice",
                    "steps": [
                        "Create invoice (account.move)",
                        "Post invoice (method: action_post)",
                        "Register payment"
                    ],
                    "model": "account.move"
                }]
            }

        # Get server actions
        custom_automations = []
        try:
            server_actions = odoo_client.search_read(
                'ir.actions.server',
                [('binding_model_id', '!=', False)],
                fields=['name', 'model_id', 'binding_model_id', 'state'],
                limit=100
            )
            for action in server_actions:
                model_id = action.get('model_id')
                model_name = model_id[1] if isinstance(model_id, (list, tuple)) else str(model_id or '')
                custom_automations.append({
                    "name": action.get('name'),
                    "model": model_name,
                    "state": action.get('state')
                })
        except Exception:
            pass

        return json.dumps({
            "installed_modules": list(module_names.keys()),
            "available_workflows": workflows,
            "custom_automations": custom_automations,
            "note": "Use execute_method to call workflow methods"
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource(
    "odoo://server/info",
    description="Get Odoo server information",
)
def get_server_info() -> str:
    """Get Odoo server metadata"""
    odoo_client = get_odoo_client()
    try:
        base_module = odoo_client.search_read(
            'ir.module.module',
            [['state', '=', 'installed'], ['name', '=', 'base']],
            fields=['latest_version'],
            limit=1
        )

        modules = odoo_client.search_read(
            'ir.module.module',
            [['state', '=', 'installed']],
            fields=['name', 'shortdesc', 'application'],
            limit=500
        )

        return json.dumps({
            "database": odoo_client.db,
            "odoo_version": base_module[0].get('latest_version', 'unknown') if base_module else 'unknown',
            "installed_modules_count": len(modules),
            "applications": [m['name'] for m in modules if m.get('application')],
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.resource(
    "odoo://domain-syntax",
    description="Complete domain operator reference with examples for building search filters",
)
def get_domain_syntax() -> str:
    """Get comprehensive domain syntax documentation."""
    domain_syntax = MODULE_KNOWLEDGE.get("domain_syntax", {})
    return json.dumps({
        "title": "Odoo Domain Syntax Reference",
        "usage": "Use with execute_method search/search_read/search_count in the domain parameter",
        **domain_syntax
    }, indent=2)


@mcp.resource(
    "odoo://model-limitations",
    description="Known model limitations and workarounds for problematic models (static + runtime-detected)",
)
def get_model_limitations() -> str:
    """Get known model limitations and workarounds, including runtime-detected issues with categorization."""
    # Static limitations from module_knowledge.json
    static_limitations = MODULE_KNOWLEDGE.get("model_limitations", {})
    static_limitations = {k: v for k, v in static_limitations.items() if not k.startswith("_")}

    result = {
        "title": "Known Model Limitations",
        "description": "Models with known issues and recommended workarounds",
        "note": "The MCP server automatically applies fallbacks, categorizes errors, and suggests solutions",
        "error_categories": {cat: {"cause": info["cause"], "solutions": info["solutions"]}
                           for cat, info in ERROR_CATEGORIES.items() if cat != "unknown"},
        "static_models": {},
        "runtime_detected": {},
        "patterns_summary": {}
    }

    # Add static (verified) limitations
    for model, issues in static_limitations.items():
        result["static_models"][model] = {
            "source": "module_knowledge.json (verified)",
            "methods": {}
        }
        for method, info in issues.items():
            result["static_models"][model]["methods"][method] = {
                "status": info.get("status", "unknown"),
                "workaround": info.get("workaround"),
                "notes": info.get("notes"),
                "verified": info.get("verified", False)
            }

    # Add runtime-detected limitations with full categorization
    all_domain_patterns = {}
    all_categories = {}

    with _RUNTIME_ISSUES_LOCK:
        runtime_snapshot = {k: dict(v) for k, v in RUNTIME_MODEL_ISSUES.items()}

    for model, methods in runtime_snapshot.items():
        result["runtime_detected"][model] = {
            "source": "runtime detection",
            "first_seen": None,
            "total_occurrences": 0,
            "methods": {}
        }

        for method, info in methods.items():
            result["runtime_detected"][model]["first_seen"] = info.get("first_seen")
            result["runtime_detected"][model]["total_occurrences"] += info.get("total_count", 0)

            method_data = {
                "total_count": info.get("total_count", 0),
                "last_seen": info.get("last_seen"),
                "by_category": {}
            }

            # Process each error category
            for category, cat_info in info.get("categories", {}).items():
                method_data["by_category"][category] = {
                    "count": cat_info.get("count", 0),
                    "cause": ERROR_CATEGORIES.get(category, {}).get("cause", "Unknown"),
                    "domain_patterns_detected": cat_info.get("domain_patterns", {}),
                    "solutions": cat_info.get("solutions", []),
                    "sample_errors": cat_info.get("sample_errors", [])[:2]  # Only 2 samples
                }

                # Aggregate patterns for summary
                all_categories[category] = all_categories.get(category, 0) + cat_info.get("count", 0)
                for pattern, count in cat_info.get("domain_patterns", {}).items():
                    all_domain_patterns[pattern] = all_domain_patterns.get(pattern, 0) + count

            result["runtime_detected"][model]["methods"][method] = method_data

    # Patterns summary - helps identify global issues
    result["patterns_summary"] = {
        "by_error_category": all_categories,
        "by_domain_pattern": all_domain_patterns,
        "recommendations": []
    }

    # Generate global recommendations based on patterns
    if all_domain_patterns.get("dot_notation", 0) > 2:
        result["patterns_summary"]["recommendations"].append(
            "Multiple dot notation issues detected. Consider querying related models separately."
        )
    if all_categories.get("timeout", 0) > 2:
        result["patterns_summary"]["recommendations"].append(
            "Multiple timeout issues detected. Consider adding database indexes or reducing query complexity."
        )
    if all_categories.get("relational_filter", 0) > 2:
        result["patterns_summary"]["recommendations"].append(
            "Relational filter issues common. Use IDs from separate queries instead of dot notation."
        )

    # Summary counts
    result["summary"] = {
        "static_models_count": len(result["static_models"]),
        "runtime_detected_count": len(result["runtime_detected"]),
        "total_models_with_issues": len(set(result["static_models"].keys()) | set(result["runtime_detected"].keys())),
        "total_runtime_occurrences": sum(
            m.get("total_occurrences", 0) for m in result["runtime_detected"].values()
        )
    }

    return json.dumps(result, indent=2)


@mcp.resource(
    "odoo://pagination",
    description="Guide for paginating large result sets with offset/limit",
)
def get_pagination_guide() -> str:
    """Get pagination patterns for execute_method."""
    pagination = MODULE_KNOWLEDGE.get("pagination", {})
    return json.dumps({
        "title": "Pagination Guide for execute_method",
        **pagination,
        "example_with_total": {
            "description": "Get page 2 of 50 records with total count",
            "step1_count": {
                "method": "search_count",
                "kwargs_json": '{"domain": [["is_company", "=", true]]}'
            },
            "step2_fetch": {
                "method": "search_read",
                "kwargs_json": '{"domain": [["is_company", "=", true]], "fields": ["name"], "limit": 50, "offset": 50, "order": "name asc"}'
            }
        }
    }, indent=2)


@mcp.resource(
    "odoo://hierarchical",
    description="Guide for querying parent/child tree structures (categories, locations, etc.)",
)
def get_hierarchical_guide() -> str:
    """Get hierarchical query patterns."""
    hierarchical = MODULE_KNOWLEDGE.get("hierarchical_models", {})
    return json.dumps({
        "title": "Hierarchical Query Guide",
        "description": "Patterns for working with parent/child tree structures in Odoo",
        **hierarchical,
        "execute_method_examples": {
            "get_category_tree": {
                "description": "Get a category and all its descendants",
                "call": "execute_method('product.category', 'search_read', kwargs_json='{\"domain\": [[\"id\", \"child_of\", 1]], \"fields\": [\"name\", \"parent_id\", \"parent_path\"]}')"
            },
            "get_ancestors": {
                "description": "Get all ancestors of a record",
                "call": "execute_method('product.category', 'search_read', kwargs_json='{\"domain\": [[\"id\", \"parent_of\", 10]], \"fields\": [\"name\", \"parent_id\"]}')"
            },
            "get_root_categories": {
                "description": "Get top-level categories only",
                "call": "execute_method('product.category', 'search_read', kwargs_json='{\"domain\": [[\"parent_id\", \"=\", false]], \"fields\": [\"name\", \"child_id\"]}')"
            },
            "get_direct_children": {
                "description": "Get immediate children of a parent",
                "call": "execute_method('hr.department', 'search_read', kwargs_json='{\"domain\": [[\"parent_id\", \"=\", 5]], \"fields\": [\"name\", \"manager_id\"]}')"
            }
        }
    }, indent=2)


@mcp.resource(
    "odoo://aggregation",
    description="Guide for aggregation: formatted_read_group (v19+) and read_group (deprecated)",
)
def get_aggregation_guide() -> str:
    """Get aggregation reference for both formatted_read_group and read_group."""
    aggregation = MODULE_KNOWLEDGE.get("aggregation", {})
    return json.dumps({
        "title": "Aggregation Guide",
        "recommendation": "Use formatted_read_group for new code (v19+). read_group still works but is deprecated.",
        **aggregation,
        "execute_method_examples": {
            "_note": "Examples using both methods. Prefer formatted_read_group for new code.",
            "formatted_read_group_examples": {
                "sales_by_customer": {
                    "description": "Total sales amount by customer (v19+ recommended)",
                    "call": "execute_method('sale.order', 'formatted_read_group', kwargs_json='{\"domain\": [], \"groupby\": [\"partner_id\"], \"aggregates\": [\"amount_total:sum\"]}')"
                },
                "invoices_by_month": {
                    "description": "Invoice count grouped by month",
                    "call": "execute_method('account.move', 'formatted_read_group', kwargs_json='{\"domain\": [[\"move_type\", \"=\", \"out_invoice\"]], \"groupby\": [\"invoice_date:month\"], \"aggregates\": [\"__count\"]}')"
                },
                "products_by_category": {
                    "description": "Product count and total value by category",
                    "call": "execute_method('product.product', 'formatted_read_group', kwargs_json='{\"domain\": [], \"groupby\": [\"categ_id\"], \"aggregates\": [\"__count\", \"list_price:sum\"]}')"
                },
                "leads_by_stage": {
                    "description": "Expected revenue by CRM stage",
                    "call": "execute_method('crm.lead', 'formatted_read_group', kwargs_json='{\"domain\": [[\"type\", \"=\", \"opportunity\"]], \"groupby\": [\"stage_id\"], \"aggregates\": [\"expected_revenue:sum\", \"__count\"]}')"
                },
                "multi_level_grouping": {
                    "description": "Sales by customer and state",
                    "call": "execute_method('sale.order', 'formatted_read_group', kwargs_json='{\"domain\": [], \"groupby\": [\"partner_id\", \"state\"], \"aggregates\": [\"amount_total:sum\"]}')"
                }
            },
            "read_group_examples_legacy": {
                "_note": "read_group is deprecated in v19 but still works. Use formatted_read_group above for new code.",
                "sales_by_customer": {
                    "description": "Total sales amount by customer (legacy)",
                    "call": "execute_method('sale.order', 'read_group', args_json='[[]]', kwargs_json='{\"fields\": [\"amount_total:sum\"], \"groupby\": [\"partner_id\"]}')"
                },
                "invoices_by_month": {
                    "description": "Invoice count grouped by month (legacy)",
                    "call": "execute_method('account.move', 'read_group', args_json='[[[\"move_type\", \"=\", \"out_invoice\"]]]', kwargs_json='{\"fields\": [\"__count\"], \"groupby\": [\"invoice_date:month\"]}')"
                }
            }
        },
        "lazy_parameter": {
            "applies_to": "read_group only (formatted_read_group always returns all levels)",
            "description": "Controls grouping behavior for multiple groupby fields",
            "default": True,
            "lazy_true": "Groups by first field only; remaining fields lazy-loaded (multiple queries)",
            "lazy_false": "All groupings in single query - more efficient for multi-level reports",
            "recommendation": "Use lazy=False when grouping by 2+ fields, or switch to formatted_read_group"
        }
    }, indent=2)


@mcp.resource(
    "odoo://find-model/{concept}",
    description="Find Odoo model from natural language concept (contact, invoice, quote, etc.)",
)
def find_model_resource(concept: str) -> str:
    """Find model from business term - converts to odoo model name."""
    odoo_client = get_odoo_client()
    concept_lower = concept.lower().strip()

    result = {
        "concept": concept,
        "best_match": None,
        "all_matches": [],
        "source": None,
    }

    # 1. Check built-in aliases first (instant)
    if concept_lower in CONCEPT_ALIASES:
        models = CONCEPT_ALIASES[concept_lower]
        result["best_match"] = models[0]
        result["all_matches"] = [{"model": m, "score": 100, "source": "alias"} for m in models]
        result["source"] = "alias"
        return json.dumps(result, indent=2)

    # 2. Search ir.model by display name
    try:
        ir_models = odoo_client.search_read(
            "ir.model",
            [["name", "ilike", concept]],
            fields=["model", "name"],
            limit=10
        )

        if ir_models:
            for m in ir_models:
                score = 90 if concept_lower == m["name"].lower() else 70
                result["all_matches"].append({
                    "model": m["model"],
                    "display_name": m["name"],
                    "score": score,
                    "source": "ir.model"
                })
            result["all_matches"].sort(key=lambda x: x["score"], reverse=True)
            result["best_match"] = result["all_matches"][0]["model"]
            result["source"] = "ir.model"
            return json.dumps(result, indent=2)
    except Exception:
        pass

    # 3. Fuzzy match
    try:
        all_models = odoo_client.search_read(
            "ir.model", [], fields=["model", "name"], limit=500
        )
        for m in all_models:
            model_name = m["model"].lower()
            display_name = m["name"].lower()
            if concept_lower in model_name or concept_lower in display_name:
                score = 80 if concept_lower in model_name.split(".") else 60
                result["all_matches"].append({
                    "model": m["model"],
                    "display_name": m["name"],
                    "score": score,
                    "source": "fuzzy"
                })

        if result["all_matches"]:
            result["all_matches"].sort(key=lambda x: x["score"], reverse=True)
            result["best_match"] = result["all_matches"][0]["model"]
            result["source"] = "fuzzy"
    except Exception as e:
        result["error"] = str(e)

    if not result["best_match"]:
        result["suggestion"] = "Check odoo://models for available models"

    return json.dumps(result, indent=2)


@mcp.resource(
    "odoo://tools/{query}",
    description="Search available operations by keyword (invoice, sales, stock, etc.)",
)
def search_tools_resource(query: str) -> str:
    """Search tool registry and module knowledge for operations."""
    query_lower = query.lower()
    matches = []

    # Search tool registry
    for tool_name, tool_def in TOOL_REGISTRY.items():
        searchable = f"{tool_name} {tool_def.get('description', '')} {tool_def.get('model', '')}".lower()
        if query_lower in searchable:
            matches.append({
                "name": tool_name,
                "type": "workflow",
                **tool_def
            })

    # Search module knowledge for special methods
    special_matches = []
    for module_name, module_info in MODULE_KNOWLEDGE.get("modules", {}).items():
        for method_name, method_info in module_info.get("special_methods", {}).items():
            searchable = f"{method_name} {method_info.get('description', '')} {module_info.get('model', '')}".lower()
            if query_lower in searchable:
                special_matches.append({
                    "name": method_name,
                    "type": "special_method",
                    "model": module_info.get("model"),
                    "description": method_info.get("description"),
                    "params": method_info.get("params", {}),
                })

    return json.dumps({
        "query": query,
        "tools_found": len(matches) + len(special_matches),
        "workflows": matches,
        "special_methods": special_matches,
        "usage": "Use execute_workflow() for workflows or execute_method() for special methods"
    }, indent=2)


@mcp.resource(
    "odoo://actions/{model}",
    description="Discover all available actions for a specific model",
)
def discover_actions_resource(model: str) -> str:
    """Discover all actions available for a model."""
    odoo_client = get_odoo_client()

    result = {
        "model": model,
        "workflows": [],
        "special_methods": [],
        "server_actions": [],
        "orm_methods": ["search", "search_read", "search_count", "create", "write", "unlink", "read", "fields_get"],
        "usage_examples": [],
    }

    # 1. Find workflows in registry for this model
    for tool_name, tool_def in TOOL_REGISTRY.items():
        if tool_def.get("model") == model:
            result["workflows"].append({
                "name": tool_name,
                "description": tool_def.get("description"),
                "params": tool_def.get("params", {}),
                "method": tool_def.get("method"),
            })

    # 2. Check module knowledge
    for module_name, module_info in MODULE_KNOWLEDGE.get("modules", {}).items():
        if module_info.get("model") == model:
            for method_name, method_info in module_info.get("special_methods", {}).items():
                result["special_methods"].append({
                    "name": method_name,
                    "description": method_info.get("description"),
                    "params": method_info.get("params", {}),
                    "replaces": method_info.get("instead_of"),
                })
                if method_info.get("instead_of"):
                    result["warnings"] = result.get("warnings", [])
                    result["warnings"].append(
                        f"Use {method_name}() instead of {method_info['instead_of']}()"
                    )
            if module_info.get("notes"):
                result["notes"] = module_info["notes"]

    # 3. Get server actions from Odoo
    try:
        actions = odoo_client.search_read(
            'ir.actions.server',
            [('model_id.model', '=', model)],
            fields=['name', 'state'],
            limit=20
        )
        for action in actions:
            result["server_actions"].append({
                "name": action.get('name'),
                "type": action.get('state'),
            })
    except Exception:
        pass

    # 4. Add usage examples
    result["usage_examples"] = [
        {
            "description": "Search records",
            "code": f'execute_method("{model}", "search_read", kwargs_json=\'{{"domain": [], "limit": 10}}\')'
        },
        {
            "description": "Create record",
            "code": f'execute_method("{model}", "create", args_json=\'[{{"field": "value"}}]\')'
        }
    ]

    if result["special_methods"]:
        method = result["special_methods"][0]
        result["usage_examples"].insert(0, {
            "description": f"Call {method['name']}",
            "code": f'execute_method("{model}", "{method["name"]}", args_json="[[id]]")'
        })

    return json.dumps(result, indent=2)


@mcp.resource(
    "odoo://tool-registry",
    description="Registry of pre-built tools and workflows (Code-First Pattern)",
)
def get_tool_registry() -> str:
    """Get the complete tool registry for code-first pattern."""
    return json.dumps({
        "description": "Pre-built tools and workflows for common Odoo operations",
        "usage": "Read odoo://tools/{query} to find tools, execute_workflow(name, params) to run",
        "tools": TOOL_REGISTRY,
        "categories": {
            "sales": [k for k, v in TOOL_REGISTRY.items() if "sale" in v.get("model", "")],
            "accounting": [k for k, v in TOOL_REGISTRY.items() if "account" in v.get("model", "")],
            "crm": [k for k, v in TOOL_REGISTRY.items() if "crm" in v.get("model", "")],
            "stock": [k for k, v in TOOL_REGISTRY.items() if "stock" in v.get("model", "")],
            "hr": [k for k, v in TOOL_REGISTRY.items() if "hr" in v.get("model", "")],
        }
    }, indent=2)
