"""
MCP Server for Odoo 19+

Provides MCP tools and resources for interacting with Odoo ERP via JSON-2 API.

MCP 2025-11-25 Features:
- Background Tasks (SEP-1686) - Async operations with progress tracking
- Icons (SEP-973) - Visual icons for server and components
- Structured Output Schemas - Typed Pydantic responses
- User Elicitation - Interactive configuration
"""

import asyncio
import base64
import json
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Progress
from fastmcp.prompts import Message
from mcp.types import Icon
from pydantic import BaseModel, Field

from .odoo_client import OdooClient, get_odoo_client


# ----- Icon Loading -----

def _load_icon() -> Optional[Icon]:
    """Load the Odoo icon from assets as a data URI."""
    icon_path = Path(__file__).parent / "assets" / "odoo_icon.svg"
    try:
        if icon_path.exists():
            icon_data = base64.standard_b64encode(icon_path.read_bytes()).decode()
            return Icon(
                src=f"data:image/svg+xml;base64,{icon_data}",
                mimeType="image/svg+xml",
            )
    except Exception as e:
        print(f"Warning: Could not load icon: {e}", file=sys.stderr)
    return None


ODOO_ICON = _load_icon()


# ----- Module Knowledge Base -----

def load_module_knowledge() -> Dict[str, Any]:
    """Load the module knowledge base from JSON file."""
    knowledge_path = Path(__file__).parent / "module_knowledge.json"
    try:
        with open(knowledge_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load module knowledge: {e}", file=sys.stderr)
        return {"modules": {}, "error_patterns": {}}


MODULE_KNOWLEDGE = load_module_knowledge()

# Runtime tracking of models that triggered fallback mechanisms
# Structure: {"model.name": {"method": {error_category: {...}}}}
RUNTIME_MODEL_ISSUES: Dict[str, Dict[str, Dict[str, Any]]] = {}

# Error categorization patterns and their solutions
ERROR_CATEGORIES = {
    "timeout": {
        "patterns": ["timeout", "statement timeout", "canceling statement", "took too long"],
        "cause": "Query too complex or slow",
        "solutions": [
            "Reduce limit parameter",
            "Simplify domain (remove complex joins)",
            "Use read_group for aggregation instead",
            "Add database indexes on filtered fields"
        ]
    },
    "relational_filter": {
        "patterns": ["relation", "join", "does not exist", "invalid field"],
        "cause": "Relational/dot notation filter issue",
        "solutions": [
            "Avoid dot notation in domain (e.g., partner_id.name)",
            "Query related model separately and use 'in' operator",
            "Use search+read fallback (automatic)"
        ]
    },
    "computed_field": {
        "patterns": ["compute", "depends", "_compute_", "stored=false"],
        "cause": "Computed field error during search",
        "solutions": [
            "Exclude computed fields from 'fields' parameter",
            "Use stored computed fields only",
            "Fetch computed fields in separate read() call"
        ]
    },
    "access_rights": {
        "patterns": ["access", "permission", "denied", "not allowed", "security"],
        "cause": "Insufficient permissions",
        "solutions": [
            "Check user access rights on model",
            "Verify record rules allow access",
            "Use fields the user has permission to read"
        ]
    },
    "memory": {
        "patterns": ["memory", "out of memory", "oom", "killed"],
        "cause": "Query uses too much memory",
        "solutions": [
            "Reduce limit significantly",
            "Paginate with smaller batches",
            "Remove large fields (binary, text) from fields list"
        ]
    },
    "data_integrity": {
        "patterns": ["integrity", "constraint", "null", "foreign key", "duplicate"],
        "cause": "Data integrity issue in database",
        "solutions": [
            "Check for orphaned records",
            "Verify foreign key references exist",
            "Contact database administrator"
        ]
    },
    "unknown": {
        "patterns": [],
        "cause": "Unknown server error",
        "solutions": [
            "Check Odoo server logs for details",
            "Try with simpler parameters",
            "Use search+read fallback (automatic)"
        ]
    }
}

# ----- /doc-bearer/ Live Documentation Cache -----

# Cache for /doc-bearer/ responses: {model_name: (timestamp, data)}
_DOC_CACHE: Dict[str, tuple] = {}
_DOC_CACHE_TTL = 300  # 5 minutes


def _strip_html(html_str: str) -> str:
    """Strip HTML tags and normalize whitespace for plain text display."""
    if not html_str:
        return ""
    text = re.sub(r'<[^>]+>', '', html_str)
    return ' '.join(text.split()).strip()


def _get_live_doc(model_name: str) -> Optional[Dict[str, Any]]:
    """Fetch live model docs from /doc-bearer/ with in-memory caching.

    Returns the doc dict if available, or None on any failure.
    Failures are silent — the caller falls back to static data.
    """
    now = time.time()
    if model_name in _DOC_CACHE:
        ts, data = _DOC_CACHE[model_name]
        if now - ts < _DOC_CACHE_TTL:
            return data

    try:
        odoo = get_odoo_client()
        doc = odoo.get_model_doc(model_name)
        if doc and isinstance(doc, dict) and "methods" in doc:
            _DOC_CACHE[model_name] = (now, doc)
            return doc
    except Exception as e:
        print(f"Warning: /doc-bearer/ unavailable for {model_name}: {e}", file=sys.stderr)

    return None


def _categorize_error(error_msg: str) -> str:
    """Categorize an error message by its root cause."""
    error_lower = error_msg.lower()
    for category, info in ERROR_CATEGORIES.items():
        if category == "unknown":
            continue
        for pattern in info["patterns"]:
            if pattern in error_lower:
                return category
    return "unknown"


def _detect_domain_pattern(domain: List, model: str = None) -> List[str]:
    """Detect patterns in domain that might cause issues."""
    patterns = []
    if not domain:
        return patterns

    domain_str = str(domain)

    # Detect dot notation (relational filters)
    if "." in domain_str and any(f".{field}" in domain_str for field in ["id", "name", "code", "state", "type", "partner", "company", "user", "product", "location"]):
        patterns.append("dot_notation")

    # Detect complex OR conditions
    if domain_str.count("'|'") > 2 or domain_str.count('"|"') > 2:
        patterns.append("complex_or")

    # Detect negation
    if "'!'" in domain_str or '"!"' in domain_str:
        patterns.append("negation")

    # Detect 'any' operator (x2many search)
    if "'any'" in domain_str or '"any"' in domain_str:
        patterns.append("any_operator")

    # Detect child_of/parent_of (hierarchical)
    if "child_of" in domain_str or "parent_of" in domain_str:
        patterns.append("hierarchical")

    # Model-specific patterns
    if model == "stock.move.line":
        # picking_type_id with negative operators causes NotImplemented error
        if "picking_type_id" in domain_str:
            if any(op in domain_str for op in ["'!='", "'not in'", "'not like'", "\"!=\"", "\"not in\""]):
                patterns.append("computed_field_negative_operator")
        # Deep related fields that cause issues
        if any(field in domain_str for field in ["product_category_name", "picking_code"]):
            patterns.append("deep_related_field")

    return patterns


def _detect_problematic_fields(fields: List, model: str = None) -> List[str]:
    """Detect fields that might cause issues when included in search_read."""
    problematic = []
    if not fields:
        return problematic

    # Model-specific problematic fields (from source code analysis)
    model_problematic_fields = {
        "stock.move.line": {
            "non_stored_computed": ["lots_visible", "allowed_uom_ids"],
            "deep_related": ["product_category_name", "picking_code"],
            "computed_with_search": ["picking_type_id"]
        }
    }

    if model in model_problematic_fields:
        for category, field_list in model_problematic_fields[model].items():
            for field in field_list:
                if field in fields:
                    problematic.append(f"{field} ({category})")

    return problematic


def _track_model_issue(model: str, method: str, error_msg: str, domain: List = None, fields: List = None) -> Dict[str, Any]:
    """
    Track a model/method issue with error categorization and pattern detection.
    Returns analysis with suggested solutions.
    """
    now = datetime.now().isoformat()
    category = _categorize_error(error_msg)
    domain_patterns = _detect_domain_pattern(domain, model) if domain else []
    problematic_fields = _detect_problematic_fields(fields, model) if fields else []

    if model not in RUNTIME_MODEL_ISSUES:
        RUNTIME_MODEL_ISSUES[model] = {}

    if method not in RUNTIME_MODEL_ISSUES[model]:
        RUNTIME_MODEL_ISSUES[model][method] = {
            "categories": {},
            "first_seen": now,
            "total_count": 0
        }

    model_issues = RUNTIME_MODEL_ISSUES[model][method]
    model_issues["total_count"] += 1
    model_issues["last_seen"] = now

    # Track by category
    if category not in model_issues["categories"]:
        model_issues["categories"][category] = {
            "count": 0,
            "domain_patterns": {},
            "sample_errors": [],
            "solutions": ERROR_CATEGORIES[category]["solutions"]
        }

    cat_info = model_issues["categories"][category]
    cat_info["count"] += 1

    # Track domain patterns for this category
    for pattern in domain_patterns:
        cat_info["domain_patterns"][pattern] = cat_info["domain_patterns"].get(pattern, 0) + 1

    # Track problematic fields
    if "problematic_fields" not in cat_info:
        cat_info["problematic_fields"] = {}
    for field_info in problematic_fields:
        cat_info["problematic_fields"][field_info] = cat_info["problematic_fields"].get(field_info, 0) + 1

    # Keep last 3 unique error samples
    error_sample = error_msg[:300]
    if error_sample not in cat_info["sample_errors"]:
        cat_info["sample_errors"].append(error_sample)
        if len(cat_info["sample_errors"]) > 3:
            cat_info["sample_errors"].pop(0)

    # Log detailed info
    print(f"[MCP] Issue tracked: {model}.{method}", file=sys.stderr)
    print(f"  Category: {category} ({ERROR_CATEGORIES[category]['cause']})", file=sys.stderr)
    if domain_patterns:
        print(f"  Domain patterns: {domain_patterns}", file=sys.stderr)
    if problematic_fields:
        print(f"  Problematic fields: {problematic_fields}", file=sys.stderr)
    print(f"  Total occurrences: {model_issues['total_count']}", file=sys.stderr)

    # Get model-specific recommendations from module_knowledge
    model_specific_advice = []
    if model in MODULE_KNOWLEDGE.get("model_limitations", {}):
        model_info = MODULE_KNOWLEDGE["model_limitations"][model]
        if method in model_info:
            method_info = model_info[method]
            if "safe_fields" in method_info:
                model_specific_advice.append(f"Safe fields: {', '.join(method_info['safe_fields'][:5])}...")
            if "avoid_in_domain" in method_info:
                model_specific_advice.append(f"Avoid in domain: {', '.join(method_info['avoid_in_domain'][:3])}")

    # Return analysis for immediate use
    return {
        "category": category,
        "cause": ERROR_CATEGORIES[category]["cause"],
        "domain_patterns": domain_patterns,
        "problematic_fields": problematic_fields,
        "solutions": ERROR_CATEGORIES[category]["solutions"],
        "model_specific_advice": model_specific_advice,
        "occurrences": cat_info["count"]
    }


def get_error_suggestion(error_msg: str, model: str = None, method: str = None) -> Optional[str]:
    """Get a helpful suggestion based on error message patterns."""
    error_patterns = MODULE_KNOWLEDGE.get("error_patterns", {})

    # Check for HTTP status code patterns
    for code, info in error_patterns.items():
        if code in str(error_msg):
            if isinstance(info, dict) and "patterns" in info:
                for pattern in info["patterns"]:
                    if pattern.get("match", "").lower() in error_msg.lower():
                        return pattern.get("suggestion")
                # Return general suggestion for this code
                if "suggestion" in info:
                    return info["suggestion"]
            elif isinstance(info, dict) and "suggestion" in info:
                return info["suggestion"]

    # Check module-specific suggestions
    if model:
        for module_name, module_info in MODULE_KNOWLEDGE.get("modules", {}).items():
            if module_info.get("model") == model:
                # Check if using wrong method
                special_methods = module_info.get("special_methods", {})
                for special_method, method_info in special_methods.items():
                    if method_info.get("instead_of") == method:
                        return f"Use {special_method}() instead of {method}() for {model}. {module_info.get('notes', '')}"

    return None


# ----- Concept to Model Mappings -----

# Common concept aliases for natural language model discovery
CONCEPT_ALIASES: Dict[str, List[str]] = {
    # Contacts & Partners
    "contact": ["res.partner"],
    "customer": ["res.partner"],
    "vendor": ["res.partner"],
    "supplier": ["res.partner"],
    "company": ["res.partner", "res.company"],

    # Sales
    "quote": ["sale.order"],
    "quotation": ["sale.order"],
    "sales order": ["sale.order"],
    "order": ["sale.order", "purchase.order"],

    # Accounting
    "invoice": ["account.move"],
    "bill": ["account.move"],
    "payment": ["account.payment"],
    "journal": ["account.journal"],

    # Products
    "product": ["product.product", "product.template"],
    "item": ["product.product"],
    "article": ["knowledge.article", "product.product"],

    # HR
    "employee": ["hr.employee"],
    "department": ["hr.department"],
    "leave": ["hr.leave"],
    "expense": ["hr.expense"],

    # Project
    "task": ["project.task"],
    "project": ["project.project"],

    # CRM
    "lead": ["crm.lead"],
    "opportunity": ["crm.lead"],
    "pipeline": ["crm.lead"],

    # Stock
    "stock": ["stock.quant", "stock.move"],
    "inventory": ["stock.quant"],
    "warehouse": ["stock.warehouse"],
    "delivery": ["stock.picking"],
    "shipment": ["stock.picking"],
    "transfer": ["stock.picking"],

    # Communication
    "message": ["mail.message"],
    "channel": ["discuss.channel"],
    "chat": ["discuss.channel"],
    "note": ["mail.message"],

    # Documents
    "document": ["documents.document", "ir.attachment"],
    "attachment": ["ir.attachment"],
    "file": ["ir.attachment"],

    # Users & Access
    "user": ["res.users"],
    "group": ["res.groups"],
    "role": ["res.groups"],
}


@dataclass
class AppContext:
    """Application context for the MCP server"""
    odoo: OdooClient


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Application lifespan for initialization and cleanup"""
    odoo_client = get_odoo_client()
    try:
        yield AppContext(odoo=odoo_client)
    finally:
        pass


# Configure authentication if MCP_API_KEY is set
def _get_auth_provider():
    """Get auth provider if MCP_API_KEY is configured."""
    api_key = os.environ.get("MCP_API_KEY")
    if api_key:
        from fastmcp.server.auth import StaticTokenVerifier
        return StaticTokenVerifier(
            tokens={
                api_key: {
                    "client_id": "mcp-client",
                    "scopes": ["read", "write"],
                }
            }
        )
    return None


# Create MCP server with icon and website URL
_auth = _get_auth_provider()
_icons = [ODOO_ICON] if ODOO_ICON else None

mcp = FastMCP(
    "Odoo 19+ MCP Server",
    lifespan=app_lifespan,
    auth=_auth,
    website_url="https://github.com/AlanOgic/odoo-mcp-19",
    icons=_icons,
)


# ----- Response Models (Structured Output Schemas) -----


class IssueAnalysis(BaseModel):
    """Analysis of issues encountered during execution."""
    category: str = Field(description="Error category: timeout, relational_filter, computed_field, access_rights, memory, data_integrity, unknown")
    cause: str = Field(description="Human-readable cause description")
    domain_patterns: List[str] = Field(default_factory=list, description="Detected patterns in domain that may cause issues")
    problematic_fields: List[str] = Field(default_factory=list, description="Fields that may cause issues")
    suggested_solutions: List[str] = Field(default_factory=list, description="Suggested solutions for the issue")
    model_specific_advice: List[str] = Field(default_factory=list, description="Model-specific recommendations")


class ExecuteMethodResponse(BaseModel):
    """Response model for execute_method tool with structured output."""
    success: bool = Field(description="Whether the execution was successful")
    result: Optional[Any] = Field(default=None, description="Result of the method call")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    suggestion: Optional[str] = Field(default=None, description="Helpful suggestion for fixing the error")
    hint: Optional[str] = Field(default=None, description="Additional hint for troubleshooting")
    fallback_used: bool = Field(default=False, description="Whether automatic fallback was triggered")
    issue_analysis: Optional[IssueAnalysis] = Field(default=None, description="Issue analysis when fallback was used")
    note: Optional[str] = Field(default=None, description="Additional note about the execution")
    execution_time_ms: Optional[float] = Field(default=None, description="Execution time in milliseconds")


class BatchOperationResult(BaseModel):
    """Result of a single batch operation."""
    operation_index: int = Field(description="Index of the operation in the batch")
    success: bool = Field(description="Whether this operation succeeded")
    result: Optional[Any] = Field(default=None, description="Result if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BatchExecuteResponse(BaseModel):
    """Response model for batch_execute tool with structured output."""
    success: bool = Field(description="Whether all operations succeeded")
    results: List[BatchOperationResult] = Field(description="Results for each operation")
    total_operations: int = Field(description="Total operations attempted")
    successful_operations: int = Field(description="Successful operations count")
    failed_operations: int = Field(description="Failed operations count")
    error: Optional[str] = Field(default=None, description="Overall error message if any operation failed")
    execution_time_ms: Optional[float] = Field(default=None, description="Total execution time in milliseconds")


class WorkflowStepResult(BaseModel):
    """Result of a single workflow step."""
    step: str = Field(description="Name of the workflow step")
    success: bool = Field(description="Whether this step succeeded")
    skipped: bool = Field(default=False, description="Whether this step was skipped")
    reason: Optional[str] = Field(default=None, description="Reason for skipping or failure")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    result: Optional[Any] = Field(default=None, description="Step result data")


class ExecuteWorkflowResponse(BaseModel):
    """Response model for execute_workflow tool with structured output."""
    workflow: str = Field(description="Name of the executed workflow")
    success: bool = Field(description="Whether the workflow completed successfully")
    steps: List[WorkflowStepResult] = Field(default_factory=list, description="Results for each workflow step")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    available_workflows: Optional[List[str]] = Field(default=None, description="Available workflows if unknown workflow requested")
    tip: Optional[str] = Field(default=None, description="Helpful tip for using workflows")
    # Additional result fields for specific workflows
    invoice_id: Optional[int] = Field(default=None, description="Created invoice ID (for invoice workflows)")
    invoice_ids: Optional[List[int]] = Field(default=None, description="Created invoice IDs (for order workflows)")
    execution_time_ms: Optional[float] = Field(default=None, description="Total execution time in milliseconds")


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
    odoo_client = get_odoo_client()
    try:
        fields = odoo_client.get_model_fields(model_name)

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
    Much smaller than /schema (~5-10KB vs 300KB).
    """
    odoo_client = get_odoo_client()
    try:
        fields = odoo_client.get_model_fields(model_name)
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


# ----- Standalone Helper Functions (callable directly) -----


def _get_module_knowledge() -> str:
    """Get the full module knowledge base with special methods and error patterns"""
    return json.dumps(MODULE_KNOWLEDGE, indent=2)


def _get_module_knowledge_by_name(module_name: str) -> str:
    """Get specific module knowledge"""
    modules = MODULE_KNOWLEDGE.get("modules", {})
    if module_name in modules:
        return json.dumps({
            "module": module_name,
            **modules[module_name]
        }, indent=2)
    else:
        available = list(modules.keys())
        return json.dumps({
            "error": f"Module '{module_name}' not found in knowledge base",
            "available_modules": available
        }, indent=2)


def _get_documentation_urls(target: str) -> str:
    """
    Get documentation URLs for a model or module.

    Returns:
    - Official docs URL
    - GitHub source URL
    - Suggested search queries
    - Special methods if known
    """
    ODOO_VERSION = "19.0"
    DOC_BASE = f"https://www.odoo.com/documentation/{ODOO_VERSION}"
    GITHUB_BASE = f"https://github.com/odoo/odoo/tree/{ODOO_VERSION}/addons"

    module_docs = MODULE_KNOWLEDGE.get("module_documentation", {})
    model_to_module = MODULE_KNOWLEDGE.get("model_to_module", {})
    modules_info = MODULE_KNOWLEDGE.get("modules", {})

    # Determine if target is a model or module
    is_model = "." in target and target not in module_docs

    if is_model:
        # It's a model - find its module
        module = model_to_module.get(target, target.split(".")[0])
        class_name = "".join(word.capitalize() for word in target.replace(".", "_").split("_"))
        filename = "_".join(target.split(".")) + ".py"
    else:
        module = target
        class_name = None
        filename = None

    result = {
        "target": target,
        "type": "model" if is_model else "module",
        "module": module,
        "documentation": {},
        "github": {},
        "search_queries": [],
        "special_methods": []
    }

    # Get module documentation
    if module in module_docs:
        mod_info = module_docs[module]
        if mod_info.get("docs"):
            result["documentation"]["user_guide"] = f"{DOC_BASE}{mod_info['docs']}.html"
        if mod_info.get("github"):
            result["github"]["models"] = f"{GITHUB_BASE}{mod_info['github']}"
        if mod_info.get("snippets"):
            result["github"]["snippets"] = f"{GITHUB_BASE}{mod_info['snippets']}"

    # Common developer docs
    result["documentation"]["orm_reference"] = f"{DOC_BASE}/developer/reference/backend/orm.html"
    result["documentation"]["actions"] = f"{DOC_BASE}/developer/reference/backend/actions.html"

    # GitHub module root
    result["github"]["module_root"] = f"{GITHUB_BASE}/{module}"

    if filename:
        result["github"]["model_file_guess"] = f"{GITHUB_BASE}/{module}/models/{filename}"

    # Search queries
    result["search_queries"] = [
        f"site:github.com/odoo/odoo/blob/{ODOO_VERSION} {target}",
        f"site:odoo.com/documentation/{ODOO_VERSION} {module}",
    ]
    if class_name:
        result["search_queries"].append(f'site:github.com/odoo/odoo "class {class_name}"')

    # Include special methods if known
    for mod_name, mod_data in modules_info.items():
        if is_model and mod_data.get("model") == target:
            result["special_methods"] = list(mod_data.get("special_methods", {}).keys())
            if mod_data.get("notes"):
                result["notes"] = mod_data["notes"]
            break
        elif not is_model and mod_name == target:
            result["special_methods"] = list(mod_data.get("special_methods", {}).keys())
            if mod_data.get("notes"):
                result["notes"] = mod_data["notes"]
            break

    return json.dumps(result, indent=2)


# ----- MCP Resources (wrap helper functions) -----


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
    description="Mapping of business concepts to Odoo model names (contact→res.partner, invoice→account.move)",
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

    for model, methods in RUNTIME_MODEL_ISSUES.items():
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
            if module_info.get("notes"):
                result["notes"] = module_info["notes"]
            if method_info.get("instead_of"):
                result["warnings"] = result.get("warnings", [])
                result["warnings"].append(f"Use {method_name}() instead of {method_info['instead_of']}()")

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


# ----- MCP Tools (Only 3: execute_method, batch_execute, execute_workflow) -----

# Icon list for tools (reusable)
_tool_icons = [ODOO_ICON] if ODOO_ICON else None


@mcp.tool(
    description="""Execute ANY Odoo method on ANY model.

    This is the universal tool for full Odoo API access.

    BEFORE USING: Read these resources for guidance:
    - odoo://actions/{model} - Discover available actions
    - odoo://methods/{model} - Method signatures
    - odoo://domain-syntax - Domain filter reference
    - odoo://aggregation - read_group guide

    MANDATORY WORKFLOW (no guessing!):
    1. FIRST: Read odoo://model/{model}/schema to get exact field names/types
    2. THEN: Build your query using schema field names
    Never guess field names - introspect schema first to avoid failed requests.

    Common patterns:
    - search_read: kwargs_json='{"domain": [...], "fields": [...], "limit": 100}'
    - create: args_json='[{"field": "value"}]'
    - write: args_json='[[ids], {"field": "value"}]'
    - unlink: args_json='[[ids]]'
    - One2many: (0,0,{}) create, (1,id,{}) update, (2,id,0) delete

    CRITICAL: Many2one fields = ALWAYS numeric ID, never the name!

    Smart limits: Default 100, Max 1000 records

    DISCOVERY RESOURCES (read these before querying):
    - odoo://model/{model}/schema - Field names & types (e.g. odoo://model/sale.order/schema)
    - odoo://model/{model}/fields - Lightweight field list (e.g. odoo://model/res.partner/fields)
    - odoo://methods/{model} - Available methods (e.g. odoo://methods/crm.lead)
    - odoo://actions/{model} - Discover actions (e.g. odoo://actions/sale.order)
    - odoo://model/{model}/docs - Rich docs with help text
    - odoo://record/{model}/{id} - Read a record
    - odoo://find-model/{concept} - Natural language lookup
    - odoo://tools/{query} - Search operations
    - odoo://docs/{target} - Documentation URLs
    - odoo://module-knowledge/{name} - Module-specific methods
    """,
    annotations={
        "title": "Execute Odoo Method",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    },
    icons=_tool_icons,
)
def execute_method(
    ctx: Context,
    model: str,
    method: str,
    args_json: str = None,
    kwargs_json: str = None,
) -> ExecuteMethodResponse:
    """
    Execute any method on an Odoo model.

    Parameters:
        model: Model name (e.g., 'res.partner')
        method: Method name (e.g., 'search_read', 'create')
        args_json: JSON array of positional arguments
        kwargs_json: JSON object of keyword arguments

    Examples:
        Search partners:
            model='res.partner'
            method='search_read'
            args_json='[[["is_company", "=", true]]]'
            kwargs_json='{"fields": ["name", "email"], "limit": 10}'

        Create record:
            model='res.partner'
            method='create'
            args_json='[{"name": "Test Company"}]'
    """
    start_time = time.time()
    odoo = ctx.request_context.lifespan_context.odoo

    try:
        args = []
        kwargs = {}

        if args_json:
            try:
                args = json.loads(args_json)
                if not isinstance(args, list):
                    return ExecuteMethodResponse(success=False, error="args_json must be a JSON array")
            except json.JSONDecodeError as e:
                return ExecuteMethodResponse(success=False, error=f"Invalid args_json: {e}")

        if kwargs_json:
            try:
                kwargs = json.loads(kwargs_json)
                if not isinstance(kwargs, dict):
                    return ExecuteMethodResponse(success=False, error="kwargs_json must be a JSON object")
            except json.JSONDecodeError as e:
                return ExecuteMethodResponse(success=False, error=f"Invalid kwargs_json: {e}")

        # Known @api.private methods that cannot be called via RPC in Odoo 19
        PRIVATE_METHOD_HINTS = {
            "check_access": "check_access is @api.private in v19. Use has_access(operation) instead (returns boolean).",
            "_read_group": "_read_group is @api.private. Use formatted_read_group (v19+) or read_group (deprecated but still works).",
            "search_fetch": "search_fetch is @api.private. Use search_read instead.",
            "fetch": "fetch is @api.private. Use read instead.",
        }

        # Static fallback check for known private methods
        if method in PRIVATE_METHOD_HINTS:
            elapsed_ms = (time.time() - start_time) * 1000
            return ExecuteMethodResponse(
                success=False,
                error=f"Method '{method}' is @api.private and cannot be called via RPC.",
                hint=PRIVATE_METHOD_HINTS[method],
                execution_time_ms=round(elapsed_ms, 2),
            )

        # Dynamic check: if method starts with _ and live doc confirms it's not public
        if method.startswith("_"):
            live_doc = _get_live_doc(model)
            if live_doc and method not in live_doc.get("methods", {}):
                elapsed_ms = (time.time() - start_time) * 1000
                return ExecuteMethodResponse(
                    success=False,
                    error=f"Method '{method}' is not a public method on {model}. It may be @api.private or doesn't exist.",
                    hint=f"Use odoo://methods/{model} to see available public methods.",
                    execution_time_ms=round(elapsed_ms, 2),
                )

        # Apply smart limits for search methods
        DEFAULT_LIMIT = 100
        MAX_LIMIT = 1000

        if method in ["search", "search_read"] and 'limit' not in kwargs:
            kwargs['limit'] = DEFAULT_LIMIT
            print(f"Applied default limit={DEFAULT_LIMIT}", file=sys.stderr)
        elif method in ["search", "search_read"] and kwargs.get('limit', 0) > MAX_LIMIT:
            kwargs['limit'] = MAX_LIMIT
            print(f"Capped limit to {MAX_LIMIT}", file=sys.stderr)

        # Normalize domain if needed
        if method in ['search', 'search_read', 'search_count'] and args:
            domain = args[0]
            # Handle double-wrapped domains [[domain]]
            if isinstance(domain, list) and len(domain) == 1 and isinstance(domain[0], list):
                if domain[0] and isinstance(domain[0][0], list):
                    args[0] = domain[0]

        result = odoo.execute_method(model, method, *args, **kwargs)
        elapsed_ms = (time.time() - start_time) * 1000
        return ExecuteMethodResponse(success=True, result=result, execution_time_ms=round(elapsed_ms, 2))

    except Exception as e:
        error_msg = str(e)

        # Fallback for search_read failures (500 errors): try search + read
        if method == "search_read" and ("500" in error_msg or "Internal Server Error" in error_msg):
            try:
                # Extract parameters from kwargs
                domain = kwargs.get("domain", [])
                fields = kwargs.get("fields", [])
                limit = kwargs.get("limit", 100)
                offset = kwargs.get("offset", 0)
                order = kwargs.get("order")

                # Step 1: search for IDs
                search_kwargs = {"domain": domain, "limit": limit, "offset": offset}
                if order:
                    search_kwargs["order"] = order
                ids = odoo.execute_method(model, "search", **search_kwargs)

                # Step 2: read the records
                if ids:
                    read_kwargs = {}
                    if fields:
                        read_kwargs["fields"] = fields
                    result = odoo.execute_method(model, "read", ids, **read_kwargs)
                else:
                    result = []

                # Track this model/method as problematic (runtime detection)
                analysis = _track_model_issue(model, method, error_msg, domain=domain, fields=fields)

                elapsed_ms = (time.time() - start_time) * 1000
                return ExecuteMethodResponse(
                    success=True,
                    result=result,
                    fallback_used=True,
                    issue_analysis=IssueAnalysis(
                        category=analysis["category"],
                        cause=analysis["cause"],
                        domain_patterns=analysis["domain_patterns"],
                        problematic_fields=analysis.get("problematic_fields", []),
                        suggested_solutions=analysis["solutions"][:2],
                        model_specific_advice=analysis.get("model_specific_advice", []),
                    ),
                    note=f"Fallback search+read used. Cause: {analysis['cause']}",
                    execution_time_ms=round(elapsed_ms, 2),
                )
            except Exception as fallback_error:
                # If fallback also fails, include both errors
                elapsed_ms = (time.time() - start_time) * 1000
                return ExecuteMethodResponse(
                    success=False,
                    error=f"{error_msg}; Fallback also failed: {fallback_error}",
                    suggestion="Both search_read and fallback search+read failed. Check odoo://model-limitations for known issues.",
                    execution_time_ms=round(elapsed_ms, 2),
                )

        suggestion = get_error_suggestion(error_msg, model, method)
        elapsed_ms = (time.time() - start_time) * 1000

        # Auto-suggest schema introspection for field-related errors
        hint = None
        field_error_patterns = ["invalid field", "unknown field", "field_get", "keyerror", "no field", "does not exist"]
        error_lower = error_msg.lower()
        if any(p in error_lower for p in field_error_patterns):
            hint = f"Field name error detected. Read odoo://model/{model}/fields to get exact field names, or odoo://model/{model}/schema for full details."
        elif suggestion:
            hint = f"Check odoo://methods/{model} or odoo://module-knowledge for special methods"

        return ExecuteMethodResponse(
            success=False,
            error=error_msg,
            suggestion=suggestion,
            hint=hint,
            execution_time_ms=round(elapsed_ms, 2),
        )


@mcp.tool(
    description="Execute multiple Odoo operations in a batch with progress tracking",
    annotations={
        "title": "Batch Execute",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    },
    icons=_tool_icons,
    task=True,  # Enable background task execution with progress
)
async def batch_execute(
    operations: List[Dict[str, Any]],
    atomic: bool = True,
    progress: Progress = Progress(),
) -> BatchExecuteResponse:
    """
    Execute multiple operations efficiently with progress tracking.

    Parameters:
        operations: List of operations, each with:
            - model: str (required)
            - method: str (required)
            - args_json: str (optional)
            - kwargs_json: str (optional)
        atomic: If True, fail fast on first error
    """
    start_time = time.time()
    # Get Odoo client directly (works in both sync and background task modes)
    odoo = get_odoo_client()
    results: List[BatchOperationResult] = []
    successful = 0
    failed = 0

    # Set up progress tracking
    await progress.set_total(len(operations))

    try:
        for idx, op in enumerate(operations):
            model = op.get('model', 'unknown')
            method = op.get('method', 'unknown')
            await progress.set_message(f"Operation {idx + 1}/{len(operations)}: {model}.{method}")

            try:
                if not op.get('model') or not op.get('method'):
                    raise ValueError(f"Operation {idx}: 'model' and 'method' required")

                args_json = op.get('args_json')
                kwargs_json = op.get('kwargs_json')

                args = json.loads(args_json) if args_json else []
                kwargs = json.loads(kwargs_json) if kwargs_json else {}

                result = odoo.execute_method(model, method, *args, **kwargs)
                results.append(BatchOperationResult(operation_index=idx, success=True, result=result))
                successful += 1

            except Exception as e:
                results.append(BatchOperationResult(operation_index=idx, success=False, error=str(e)))
                failed += 1

                if atomic:
                    elapsed_ms = (time.time() - start_time) * 1000
                    return BatchExecuteResponse(
                        success=False,
                        results=results,
                        total_operations=len(operations),
                        successful_operations=successful,
                        failed_operations=failed,
                        error=f"Failed at operation {idx}: {e}",
                        execution_time_ms=round(elapsed_ms, 2),
                    )

            await progress.increment()
            # Small delay to allow progress updates to propagate
            await asyncio.sleep(0.01)

        elapsed_ms = (time.time() - start_time) * 1000
        return BatchExecuteResponse(
            success=failed == 0,
            results=results,
            total_operations=len(operations),
            successful_operations=successful,
            failed_operations=failed,
            error=None if failed == 0 else f"{failed} operations failed",
            execution_time_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return BatchExecuteResponse(
            success=False,
            results=results,
            total_operations=len(operations),
            successful_operations=successful,
            failed_operations=failed,
            error=str(e),
            execution_time_ms=round(elapsed_ms, 2),
        )


# ----- User Elicitation Tool -----


@dataclass
class OdooConnectionConfig:
    """Configuration collected from user elicitation."""
    url: str
    database: str
    auth_method: str
    username: str


@mcp.tool(
    description="""Interactive Odoo connection configuration using user elicitation.

    This tool guides users through setting up Odoo connection parameters
    interactively, collecting URL, database, and authentication details.

    Note: This requires an MCP client that supports user elicitation.
    The collected configuration is returned but not automatically applied -
    users should set the corresponding environment variables.
    """,
    annotations={
        "title": "Configure Odoo Connection",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    },
    icons=_tool_icons,
)
async def configure_odoo(ctx: Context) -> Dict[str, Any]:
    """
    Interactive Odoo connection configuration using MCP elicitation.

    Returns:
        Configuration summary with environment variable instructions
    """
    from fastmcp.server.elicitation import AcceptedElicitation, DeclinedElicitation, CancelledElicitation

    results = {
        "success": False,
        "config": {},
        "env_vars": {},
    }

    try:
        # Step 1: Ask for Odoo URL
        url_result = await ctx.elicit(
            message="Enter your Odoo server URL (e.g., https://mycompany.odoo.com):",
            response_type=str,
        )

        match url_result:
            case AcceptedElicitation(data=url):
                results["config"]["url"] = url
            case DeclinedElicitation() | CancelledElicitation():
                results["error"] = "Configuration cancelled by user"
                return results

        # Step 2: Ask for database name
        db_result = await ctx.elicit(
            message="Enter the database name:",
            response_type=str,
        )

        match db_result:
            case AcceptedElicitation(data=database):
                results["config"]["database"] = database
            case DeclinedElicitation() | CancelledElicitation():
                results["error"] = "Configuration cancelled by user"
                return results

        # Step 3: Ask for authentication method
        auth_result = await ctx.elicit(
            message="Select authentication method:",
            response_type=["API Key (Recommended)", "Password"],
        )

        match auth_result:
            case AcceptedElicitation(data=auth_method):
                results["config"]["auth_method"] = "api_key" if "API" in auth_method else "password"
            case DeclinedElicitation() | CancelledElicitation():
                results["error"] = "Configuration cancelled by user"
                return results

        # Step 4: Ask for username
        user_result = await ctx.elicit(
            message="Enter your Odoo username (email):",
            response_type=str,
        )

        match user_result:
            case AcceptedElicitation(data=username):
                results["config"]["username"] = username
            case DeclinedElicitation() | CancelledElicitation():
                results["error"] = "Configuration cancelled by user"
                return results

        # Build environment variables
        results["success"] = True
        results["env_vars"] = {
            "ODOO_URL": results["config"]["url"],
            "ODOO_DB": results["config"]["database"],
            "ODOO_USERNAME": results["config"]["username"],
        }

        if results["config"]["auth_method"] == "api_key":
            results["env_vars"]["ODOO_API_KEY"] = "<your-api-key>"
            results["note"] = "Generate an API key in Odoo: Settings > Users > Preferences > API Keys"
        else:
            results["env_vars"]["ODOO_PASSWORD"] = "<your-password>"
            results["note"] = "Using password authentication. API keys are recommended for production."

        results["instructions"] = (
            "Set these environment variables to configure the Odoo MCP server:\n"
            + "\n".join(f"export {k}='{v}'" for k, v in results["env_vars"].items())
        )

        return results

    except Exception as e:
        if "elicitation is not supported" in str(e).lower():
            return {
                "success": False,
                "error": "User elicitation not supported by this MCP client",
                "alternative": "Set environment variables manually: ODOO_URL, ODOO_DB, ODOO_USERNAME, ODOO_API_KEY",
            }
        results["error"] = str(e)
        return results


# ----- MCP Prompts -----


@mcp.prompt(name="odoo-exploration")
def odoo_exploration_prompt() -> list[Message]:
    """Discover capabilities of this Odoo instance"""
    return [Message("""Explore this Odoo instance:

1. Read odoo://server/info for version and apps
2. Read odoo://workflows for available workflows
3. Read odoo://models for all models

Provide a summary of what's available.
""")]


@mcp.prompt(name="search-records")
def search_records_prompt(model: str = "res.partner") -> list[Message]:
    """Search for records in a model"""
    return [Message(f"""Search for records in {model}.

First read odoo://model/{model}/schema to understand the fields.

Then use execute_method with:
- model='{model}'
- method='search_read'
- args_json='[[]]'  # empty domain for all records
- kwargs_json='{{"fields": ["name", "id"], "limit": 10}}'
""")]


@mcp.prompt(name="odoo-api-reference")
def api_reference_prompt() -> list[Message]:
    """Quick reference for Odoo API patterns"""
    return [Message("""## Odoo API Quick Reference

**Method Patterns:**
- search_read: `kwargs_json='{"domain": [...], "fields": [...], "limit": 100}'`
- create: `args_json='[{"field": "value"}]'`
- write: `args_json='[[ids], {"field": "value"}]'`
- unlink: `args_json='[[ids]]'`

**One2many/Many2many Commands:**
| Code | Meaning | Syntax |
|------|---------|--------|
| 0 | Create | `(0, 0, {values})` |
| 1 | Update | `(1, id, {values})` |
| 2 | Delete | `(2, id, 0)` |
| 3 | Unlink (M2M) | `(3, id, 0)` |
| 4 | Link (M2M) | `(4, id, 0)` |
| 5 | Unlink all (M2M) | `(5, 0, 0)` |
| 6 | Replace all (M2M) | `(6, 0, [ids])` |

**Domain Operators:**
- Comparison: `=`, `!=`, `>`, `<`, `>=`, `<=`
- List: `in`, `not in`
- Text: `like`, `ilike`, `=like`, `=ilike`
- Logic: `&` (AND), `|` (OR), `!` (NOT)

**Domain Examples:**
```python
[("state", "=", "draft")]                    # Simple
[("amount", ">", 1000)]                      # Comparison
[("name", "ilike", "%test%")]                # Text search
["&", ("state", "=", "sale"), ("amount", ">", 500)]  # AND
["|", ("state", "=", "draft"), ("state", "=", "sent")]  # OR
[("partner_id.name", "=", "Company")]        # Dot notation for related fields
```

**CRITICAL WARNINGS:**
- Many2one fields = ALWAYS use numeric ID, never the name string!
- Read odoo://actions/{model} BEFORE calling unfamiliar models
- Check odoo://docs/{model} for documentation URLs

**Pre-execution Checklist:**
1. Model identified? (use odoo://find-model/{concept})
2. Actions verified? (read odoo://actions/{model})
3. Required fields known? (read odoo://model/{model}/schema)
4. Types correct? (Many2one = ID, not name)
""")]


@mcp.prompt(name="domain-builder")
def domain_builder_prompt(description: str = "") -> list[Message]:
    """Help construct complex domain filters"""
    return [Message(f"""## Domain Builder

Build a domain filter{' for: ' + description if description else ''}.

**Read odoo://domain-syntax for complete operator reference.**

**Key Operators:**
| Operator | Purpose | Example |
|----------|---------|---------|
| `=`, `!=` | Equality | `["state", "=", "draft"]` |
| `>`, `<`, `>=`, `<=` | Comparison | `["amount", ">", 1000]` |
| `in`, `not in` | List membership | `["state", "in", ["draft", "sent"]]` |
| `ilike` | Case-insensitive search | `["email", "ilike", "@gmail"]` |
| `child_of` | Hierarchical children | `["category_id", "child_of", 5]` |
| `parent_of` | Hierarchical parents | `["id", "parent_of", 10]` |
| `any` | x2many contains match | `["order_line", "any", [["product_id", "=", 1]]]` |

**Logic (Polish notation):**
- AND: `["&", term1, term2]`
- OR: `["|", term1, term2]`
- NOT: `["!", term]`

**Dot Notation for Related Fields:**
- `["partner_id.country_id.code", "=", "US"]`

**Example: Active US/CA companies with orders > $1000:**
```python
["&", "&", ["active", "=", true], ["is_company", "=", true],
 "|", ["country_id.code", "=", "US"], ["country_id.code", "=", "CA"]]
```

Use with execute_method search_read kwargs_json.
""")]


@mcp.prompt(name="hierarchical-query")
def hierarchical_query_prompt(model: str = "product.category") -> list[Message]:
    """Guide for querying parent/child tree structures"""
    return [Message(f"""## Hierarchical Query Guide for {model}

**Read odoo://hierarchical for complete patterns.**

**Query Patterns with execute_method:**

1. **Get all descendants (children + grandchildren):**
```
execute_method('{model}', 'search_read',
  kwargs_json='{{"domain": [["id", "child_of", PARENT_ID]], "fields": ["name", "parent_id"]}}')
```

2. **Get all ancestors (parents + grandparents):**
```
execute_method('{model}', 'search_read',
  kwargs_json='{{"domain": [["id", "parent_of", CHILD_ID]], "fields": ["name", "parent_id"]}}')
```

3. **Get direct children only:**
```
execute_method('{model}', 'search_read',
  kwargs_json='{{"domain": [["parent_id", "=", PARENT_ID]], "fields": ["name"]}}')
```

4. **Get root records (no parent):**
```
execute_method('{model}', 'search_read',
  kwargs_json='{{"domain": [["parent_id", "=", false]], "fields": ["name", "child_id"]}}')
```

5. **Get tree path (if model has _parent_store):**
```
execute_method('{model}', 'search_read',
  kwargs_json='{{"domain": [["id", "=", ID]], "fields": ["name", "parent_path"]}}')
```
The parent_path field contains ancestor IDs separated by '/'.

**Common hierarchical models:**
- product.category (parent_id)
- account.account (parent_id)
- hr.department (parent_id)
- stock.location (location_id)
- knowledge.article (parent_id)
""")]


@mcp.prompt(name="paginated-search")
def paginated_search_prompt(model: str = "res.partner") -> list[Message]:
    """Guide for paginating large result sets"""
    return [Message(f"""## Paginated Search Guide for {model}

**Read odoo://pagination for complete reference.**

**Pattern: Get total count + paginated results**

Step 1: Get total count
```
execute_method('{model}', 'search_count',
  kwargs_json='{{"domain": YOUR_DOMAIN}}')
```

Step 2: Fetch page of results
```
execute_method('{model}', 'search_read',
  kwargs_json='{{
    "domain": YOUR_DOMAIN,
    "fields": ["name", "..."],
    "limit": 50,
    "offset": 0,
    "order": "name asc"
  }}')
```

**Pagination formula:**
- Page 1: offset=0, limit=50
- Page 2: offset=50, limit=50
- Page N: offset=(N-1)*limit, limit=50

**Iterate all records:**
```python
offset = 0
limit = 100
while True:
    results = execute_method('{model}', 'search_read',
      kwargs_json=f'{{"domain": [], "limit": {{limit}}, "offset": {{offset}}}}')
    if len(results) < limit:
        break  # Last page
    offset += limit
```

**Default limits:**
- MCP default: 100 records
- MCP maximum: 1000 records
- Use search_count first to know total
""")]


@mcp.prompt(name="aggregation-report")
def aggregation_report_prompt(model: str = "sale.order") -> list[Message]:
    """Guide for creating aggregation reports"""
    return [Message(f"""## Aggregation Report Guide for {model}

**Read odoo://aggregation for complete reference.**

**Using read_group with execute_method:**

```
execute_method('{model}', 'read_group',
  args_json='[DOMAIN]',
  kwargs_json='{{
    "fields": ["field:aggregator", ...],
    "groupby": ["field", ...]
  }}')
```

**Aggregators:**
| Aggregator | Purpose |
|------------|---------|
| `__count` | Count records |
| `sum` | Sum values |
| `avg` | Average |
| `min` | Minimum |
| `max` | Maximum |
| `count_distinct` | Distinct count |

**Date Grouping:**
- `:day`, `:week`, `:month`, `:quarter`, `:year`
- Example: `"groupby": ["create_date:month"]`

**Examples for {model}:**

1. Total by partner:
```
kwargs_json='{{"fields": ["amount_total:sum"], "groupby": ["partner_id"]}}'
```

2. Count by state:
```
kwargs_json='{{"fields": ["__count"], "groupby": ["state"]}}'
```

3. Monthly totals:
```
kwargs_json='{{"fields": ["amount_total:sum", "__count"], "groupby": ["date_order:month"]}}'
```

4. Multi-level grouping:
```
kwargs_json='{{"fields": ["amount_total:sum"], "groupby": ["partner_id", "state"]}}'
```

**Note:** read_group is deprecated in v19. Current MCP uses it for compatibility.
New code should use formatted_read_group (web module).
""")]


# =====================================================
# VERY HIGH IMPACT: Code-First Pattern & Dynamic Discovery
# =====================================================
# These features reduce token usage by ~98% by loading
# tool definitions on-demand instead of upfront.
# =====================================================


# ----- Tool Registry for Code-First Pattern -----

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Sales Operations
    "create_quotation": {
        "description": "Create a new sales quotation",
        "model": "sale.order",
        "workflow": ["find customer", "add products", "create order"],
        "params": {"partner_id": "int", "order_line": "list of (0,0,{product_id, qty})"},
    },
    "confirm_order": {
        "description": "Confirm a quotation to sales order",
        "model": "sale.order",
        "method": "action_confirm",
        "params": {"order_ids": "list of int"},
    },
    "create_invoice_from_order": {
        "description": "Create invoice from confirmed sales order",
        "model": "sale.order",
        "method": "_create_invoices",
        "params": {"order_ids": "list of int"},
    },

    # Accounting Operations
    "post_invoice": {
        "description": "Post/validate a draft invoice",
        "model": "account.move",
        "method": "action_post",
        "params": {"invoice_ids": "list of int"},
    },
    "register_payment": {
        "description": "Register payment for an invoice",
        "model": "account.payment",
        "workflow": ["create payment", "link to invoice", "confirm"],
    },
    "get_overdue_invoices": {
        "description": "Find invoices past due date",
        "model": "account.move",
        "domain_template": [["move_type", "=", "out_invoice"], ["payment_state", "in", ["not_paid", "partial"]], ["invoice_date_due", "<", "{today}"]],
    },
    "get_ar_aging": {
        "description": "Get accounts receivable aging report",
        "model": "account.move",
        "aggregation": True,
        "groupby": ["partner_id"],
        "fields": ["amount_residual:sum"],
    },

    # CRM Operations
    "create_lead": {
        "description": "Create a new CRM lead",
        "model": "crm.lead",
        "params": {"name": "str", "partner_name": "str", "email_from": "str"},
    },
    "convert_lead_to_opportunity": {
        "description": "Convert lead to opportunity",
        "model": "crm.lead",
        "method": "convert_opportunity",
    },
    "mark_opportunity_won": {
        "description": "Mark opportunity as won",
        "model": "crm.lead",
        "method": "action_set_won",
    },

    # Stock Operations
    "check_stock_levels": {
        "description": "Check current stock quantities",
        "model": "stock.quant",
        "aggregation": True,
        "groupby": ["product_id", "location_id"],
        "fields": ["quantity:sum"],
    },
    "validate_delivery": {
        "description": "Validate a delivery/transfer",
        "model": "stock.picking",
        "method": "button_validate",
    },

    # HR Operations
    "create_employee": {
        "description": "Create a new employee record",
        "model": "hr.employee",
        "params": {"name": "str", "department_id": "int", "job_title": "str"},
    },

    # Communication
    "send_message": {
        "description": "Post a message on a record (chatter)",
        "model": "mail.message",
        "params": {
            "model": "str (target model, e.g. 'res.partner')",
            "res_id": "int (target record ID)",
            "body": "html (message content)",
            "message_type": "str REQUIRED: 'comment'|'notification'|'email'",
            "subtype_id": "int (1=visible to followers, 2=internal note)"
        },
        "note": "Create mail.message directly. message_type is REQUIRED!",
        "example": {
            "model": "res.partner",
            "res_id": 123,
            "body": "<p>Hello!</p>",
            "message_type": "comment",
            "subtype_id": 1
        }
    },
    "create_channel": {
        "description": "Create a discuss channel",
        "model": "discuss.channel",
        "params": {"name": "str", "channel_type": "channel|chat|group"},
    },
}


@mcp.tool(
    description="""Execute a multi-step workflow in a single call with progress tracking.

    This is the KEY TOOL for Code-First Pattern - combines multiple
    operations into one call, dramatically reducing tokens.

    Supported workflows:
    - quote_to_cash: Create quote -> Confirm -> Deliver -> Invoice -> Payment
    - lead_to_won: Create lead -> Convert to opportunity -> Mark won
    - create_and_post_invoice: Create invoice -> Post it
    - stock_transfer: Create transfer -> Confirm -> Validate

    Or describe a custom workflow in natural language.
    """,
    annotations={
        "title": "Execute Workflow",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    },
    icons=_tool_icons,
    task=True,  # Enable background task execution with progress
)
async def execute_workflow(
    workflow: str,
    params_json: str = None,
    progress: Progress = Progress(),
) -> ExecuteWorkflowResponse:
    """
    Execute a multi-step workflow with progress tracking.

    Parameters:
        workflow: Workflow name or description
        params_json: JSON object with workflow parameters

    Returns:
        Results from each step of the workflow
    """
    start_time = time.time()
    # Get Odoo client directly (works in both sync and background task modes)
    odoo = get_odoo_client()

    try:
        params = json.loads(params_json) if params_json else {}
    except json.JSONDecodeError as e:
        return ExecuteWorkflowResponse(
            workflow=workflow,
            success=False,
            error=f"Invalid params_json: {e}",
        )

    workflow_lower = workflow.lower().strip()
    steps: List[WorkflowStepResult] = []

    try:
        # ----- Quote to Cash Workflow -----
        if workflow_lower in ["quote_to_cash", "quotation_to_invoice", "sales_workflow"]:
            order_id = params.get("order_id")

            if not order_id:
                return ExecuteWorkflowResponse(
                    workflow=workflow,
                    success=False,
                    error="order_id required for quote_to_cash workflow",
                )

            # 3 steps: confirm, create invoice, post invoice
            await progress.set_total(3)

            # Step 1: Confirm order
            await progress.set_message("Confirming sales order...")
            try:
                odoo.execute_method("sale.order", "action_confirm", [order_id])
                steps.append(WorkflowStepResult(step="confirm_order", success=True))
            except Exception as e:
                steps.append(WorkflowStepResult(step="confirm_order", success=False, error=str(e)))
                elapsed_ms = (time.time() - start_time) * 1000
                return ExecuteWorkflowResponse(
                    workflow=workflow,
                    success=False,
                    steps=steps,
                    execution_time_ms=round(elapsed_ms, 2),
                )
            await progress.increment()

            # Step 2: Create invoice
            await progress.set_message("Creating invoice...")
            invoice_ids = None
            try:
                invoice_ids = odoo.execute_method("sale.order", "_create_invoices", [order_id])
                steps.append(WorkflowStepResult(step="create_invoice", success=True, result={"invoice_ids": invoice_ids}))
            except Exception as e:
                steps.append(WorkflowStepResult(step="create_invoice", success=False, error=str(e)))
                elapsed_ms = (time.time() - start_time) * 1000
                return ExecuteWorkflowResponse(
                    workflow=workflow,
                    success=False,
                    steps=steps,
                    execution_time_ms=round(elapsed_ms, 2),
                )
            await progress.increment()

            # Step 3: Post invoice (optional)
            await progress.set_message("Posting invoice...")
            if params.get("post_invoice", True) and invoice_ids:
                try:
                    odoo.execute_method("account.move", "action_post", invoice_ids)
                    steps.append(WorkflowStepResult(step="post_invoice", success=True))
                except Exception as e:
                    steps.append(WorkflowStepResult(step="post_invoice", success=False, error=str(e)))
            await progress.increment()

            elapsed_ms = (time.time() - start_time) * 1000
            return ExecuteWorkflowResponse(
                workflow=workflow,
                success=all(s.success or s.skipped for s in steps),
                steps=steps,
                invoice_ids=invoice_ids,
                execution_time_ms=round(elapsed_ms, 2),
            )

        # ----- Lead to Won Workflow -----
        elif workflow_lower in ["lead_to_won", "crm_workflow", "opportunity_won"]:
            lead_id = params.get("lead_id")

            if not lead_id:
                return ExecuteWorkflowResponse(
                    workflow=workflow,
                    success=False,
                    error="lead_id required for lead_to_won workflow",
                )

            # 2 steps: convert, mark won
            await progress.set_total(2)

            # Step 1: Convert to opportunity (if still a lead)
            await progress.set_message("Converting lead to opportunity...")
            try:
                lead = odoo.search_read("crm.lead", [["id", "=", lead_id]], fields=["type"], limit=1)
                if lead and lead[0].get("type") == "lead":
                    odoo.execute_method("crm.lead", "convert_opportunity", [lead_id], partner_id=params.get("partner_id", False))
                    steps.append(WorkflowStepResult(step="convert_to_opportunity", success=True))
                else:
                    steps.append(WorkflowStepResult(step="convert_to_opportunity", success=True, skipped=True, reason="Already an opportunity"))
            except Exception as e:
                steps.append(WorkflowStepResult(step="convert_to_opportunity", success=False, error=str(e)))
            await progress.increment()

            # Step 2: Mark as won
            await progress.set_message("Marking opportunity as won...")
            try:
                odoo.execute_method("crm.lead", "action_set_won", [lead_id])
                steps.append(WorkflowStepResult(step="mark_won", success=True))
            except Exception as e:
                steps.append(WorkflowStepResult(step="mark_won", success=False, error=str(e)))
            await progress.increment()

            elapsed_ms = (time.time() - start_time) * 1000
            return ExecuteWorkflowResponse(
                workflow=workflow,
                success=all(s.success or s.skipped for s in steps),
                steps=steps,
                execution_time_ms=round(elapsed_ms, 2),
            )

        # ----- Create and Post Invoice Workflow -----
        elif workflow_lower in ["create_and_post_invoice", "quick_invoice"]:
            partner_id = params.get("partner_id")
            lines = params.get("lines", [])

            if not partner_id:
                return ExecuteWorkflowResponse(
                    workflow=workflow,
                    success=False,
                    error="partner_id required",
                )
            if not lines:
                return ExecuteWorkflowResponse(
                    workflow=workflow,
                    success=False,
                    error="lines required (list of {product_id, quantity, price_unit})",
                )

            # 2 steps: create, post
            await progress.set_total(2)

            # Build invoice lines
            invoice_lines = []
            for line in lines:
                invoice_lines.append((0, 0, {
                    "product_id": line.get("product_id"),
                    "quantity": line.get("quantity", 1),
                    "price_unit": line.get("price_unit"),
                    "name": line.get("name", "Product"),
                }))

            # Step 1: Create invoice
            await progress.set_message("Creating invoice...")
            invoice_id = None
            try:
                invoice_vals = {
                    "move_type": "out_invoice",
                    "partner_id": partner_id,
                    "invoice_line_ids": invoice_lines,
                }
                invoice_id = odoo.execute_method("account.move", "create", [invoice_vals])
                steps.append(WorkflowStepResult(step="create_invoice", success=True, result={"invoice_id": invoice_id}))
            except Exception as e:
                steps.append(WorkflowStepResult(step="create_invoice", success=False, error=str(e)))
                elapsed_ms = (time.time() - start_time) * 1000
                return ExecuteWorkflowResponse(
                    workflow=workflow,
                    success=False,
                    steps=steps,
                    execution_time_ms=round(elapsed_ms, 2),
                )
            await progress.increment()

            # Step 2: Post invoice
            await progress.set_message("Posting invoice...")
            if params.get("post", True):
                try:
                    odoo.execute_method("account.move", "action_post", [invoice_id])
                    steps.append(WorkflowStepResult(step="post_invoice", success=True))
                except Exception as e:
                    steps.append(WorkflowStepResult(step="post_invoice", success=False, error=str(e)))
            await progress.increment()

            elapsed_ms = (time.time() - start_time) * 1000
            return ExecuteWorkflowResponse(
                workflow=workflow,
                success=all(s.success or s.skipped for s in steps),
                steps=steps,
                invoice_id=invoice_id,
                execution_time_ms=round(elapsed_ms, 2),
            )

        # ----- Unknown workflow -----
        else:
            return ExecuteWorkflowResponse(
                workflow=workflow,
                success=False,
                error=f"Unknown workflow: {workflow}",
                available_workflows=[
                    "quote_to_cash - Confirm order, create & post invoice",
                    "lead_to_won - Convert lead and mark as won",
                    "create_and_post_invoice - Create and post a customer invoice",
                ],
                tip="Read odoo://tools/{query} to find available operations",
            )

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return ExecuteWorkflowResponse(
            workflow=workflow,
            success=False,
            steps=steps,
            error=str(e),
            execution_time_ms=round(elapsed_ms, 2),
        )


# =====================================================
# Business Workflow Prompts
# =====================================================


@mcp.prompt(name="quote-to-cash")
def quote_to_cash_prompt(order_id: str = None) -> list[Message]:
    """Complete quote-to-cash workflow"""
    if order_id:
        return [Message(f"""Execute the quote-to-cash workflow for order {order_id}:

Use execute_workflow("quote_to_cash", '{{"order_id": {order_id}}}')

This will:
1. Confirm the sales order (action_confirm)
2. Create invoice (_create_invoices)
3. Post invoice (action_post)

Report the result of each step.
""")]
    else:
        return [Message("""Guide me through creating a complete sales flow:

1. First, find or create a customer (res.partner)
2. Create a quotation (sale.order) with order lines
3. Confirm the quotation
4. Create and post the invoice
5. Optionally register payment

Read odoo://tools/sales for available operations.
""")]


@mcp.prompt(name="ar-aging-report")
def ar_aging_report_prompt() -> list[Message]:
    """Generate accounts receivable aging report"""
    return [Message("""Generate an AR aging report:

1. Use execute_method with read_group (see odoo://aggregation):
   execute_method("account.move", "read_group",
     args_json='[[["move_type", "=", "out_invoice"], ["payment_state", "in", ["not_paid", "partial"]]]]',
     kwargs_json='{"fields": ["amount_residual:sum"], "groupby": ["partner_id"]}')

2. Then categorize by aging buckets:
   - Current (not yet due)
   - 1-30 days overdue
   - 31-60 days overdue
   - 61-90 days overdue
   - 90+ days overdue

3. Present as a formatted table with customer totals and recommendations.
""")]


@mcp.prompt(name="inventory-check")
def inventory_check_prompt(product: str = None) -> list[Message]:
    """Check inventory levels"""
    if product:
        return [Message(f"""Check inventory for "{product}":

1. Find the product: read odoo://find-model/product, then search product.product
2. Use execute_method with read_group (see odoo://aggregation):
   execute_method("stock.quant", "read_group",
     args_json='[[["product_id", "=", PRODUCT_ID]]]',
     kwargs_json='{{"fields": ["quantity:sum"], "groupby": ["location_id"]}}')

3. Show available quantity by warehouse/location
""")]
    else:
        return [Message("""Check overall inventory status:

1. Use execute_method with read_group (see odoo://aggregation):
   execute_method("stock.quant", "read_group", args_json='[[]]',
     kwargs_json='{"fields": ["quantity:sum", "value:sum"], "groupby": ["product_id"]}')

2. Identify low stock items (quantity < reorder point)
3. Show top products by value
""")]


@mcp.prompt(name="crm-pipeline")
def crm_pipeline_prompt() -> list[Message]:
    """Analyze CRM pipeline"""
    return [Message("""Analyze the CRM pipeline:

1. Use execute_method with read_group (see odoo://aggregation):
   execute_method("crm.lead", "read_group",
     args_json='[[["type", "=", "opportunity"]]]',
     kwargs_json='{"fields": ["expected_revenue:sum", "__count"], "groupby": ["stage_id"]}')

2. Calculate conversion rates between stages
3. Identify opportunities that need attention:
   - Stuck in stage too long
   - High value opportunities
   - Upcoming activities

4. Present a pipeline summary with recommendations.
""")]


@mcp.prompt(name="customer-360")
def customer_360_prompt(customer: str) -> list[Message]:
    """Complete customer 360 view"""
    return [Message(f"""Get a 360-degree view of customer "{customer}":

1. Find the customer: read odoo://find-model/customer, then search res.partner

2. Get their data:
   - Basic info (name, email, phone, address)
   - Credit limit and receivables

3. Sales history (see odoo://aggregation):
   execute_method("sale.order", "read_group",
     args_json='[[["partner_id", "=", CUSTOMER_ID]]]',
     kwargs_json='{{"fields": ["amount_total:sum"], "groupby": ["state"]}}')

4. Invoice status:
   execute_method("account.move", "read_group",
     args_json='[[["partner_id", "=", CUSTOMER_ID], ["move_type", "=", "out_invoice"]]]',
     kwargs_json='{{"fields": ["amount_residual:sum"], "groupby": ["payment_state"]}}')

5. Recent activities:
   - Messages and notes from mail.message

6. CRM opportunities:
   - Open opportunities from crm.lead

Present a comprehensive customer profile with key insights.
""")]


@mcp.prompt(name="daily-operations")
def daily_operations_prompt() -> list[Message]:
    """Daily operations dashboard"""
    return [Message("""Generate a daily operations summary:

**Sales:**
- New orders today (use read_group on sale.order, see odoo://aggregation)
- Pending quotations needing follow-up

**Inventory:**
- Pending deliveries (stock.picking with state = assigned or waiting)
- Low stock alerts

**Accounting:**
- Invoices to send (draft invoices)
- Overdue payments (use AR aging prompt)
- Cash position

**CRM:**
- Activities due today
- Hot opportunities (high probability, high value)

Use execute_method with read_group for efficient aggregation. Present as a dashboard.
""")]


# ----- Resource for listing available workflows -----


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
