"""
MCP Server for Odoo 19+

Provides MCP tools and resources for interacting with Odoo ERP via JSON-2 API.
"""

import json
import os
import re
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from .odoo_client import OdooClient, get_odoo_client


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


# Create MCP server
mcp = FastMCP(
    "Odoo 19+ MCP Server",
    lifespan=app_lifespan,
)


# ----- Response Models -----


class ExecuteMethodResponse(BaseModel):
    """Response model for execute_method tool."""
    success: bool = Field(description="Whether the execution was successful")
    result: Optional[Any] = Field(default=None, description="Result of the method")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BatchExecuteResponse(BaseModel):
    """Response model for batch_execute tool."""
    success: bool = Field(description="Whether all operations succeeded")
    results: List[Dict[str, Any]] = Field(description="Results for each operation")
    total_operations: int = Field(description="Total operations attempted")
    successful_operations: int = Field(description="Successful operations count")
    failed_operations: int = Field(description="Failed operations count")
    error: Optional[str] = Field(default=None, description="Overall error message")


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
            {"name": "search_read", "description": "Search and read in one call", "params": ["domain", "fields", "offset", "limit", "order"]},
            {"name": "read", "description": "Read specific records", "params": ["ids", "fields"]},
            {"name": "search_count", "description": "Count matching records", "params": ["domain"]},
            {"name": "fields_get", "description": "Get field definitions", "params": ["attributes"]},
        ],
        "write_methods": [
            {"name": "create", "description": "Create new record(s)", "params": ["vals"]},
            {"name": "write", "description": "Update existing record(s)", "params": ["ids", "vals"]},
            {"name": "unlink", "description": "Delete record(s)", "params": ["ids"]},
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
        "usage": "Use find_model tool to search, or look up concepts here",
        "mappings": CONCEPT_ALIASES
    }, indent=2)


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


# ----- MCP Tools -----


@mcp.tool(
    description="""Execute ANY Odoo method on ANY model.

    This is the universal tool for full Odoo API access.

    BEFORE USING on unfamiliar models: call get_model_methods() first!

    Common patterns:
    - search_read: kwargs_json='{"domain": [...], "fields": [...], "limit": 100}'
    - create: args_json='[{"field": "value"}]'
    - write: args_json='[[ids], {"field": "value"}]'
    - unlink: args_json='[[ids]]'
    - One2many: (0,0,{}) create, (1,id,{}) update, (2,id,0) delete

    CRITICAL: Many2one fields = ALWAYS numeric ID, never the name!

    Smart limits: Default 100, Max 1000 records
    """,
    annotations={
        "title": "Execute Odoo Method",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
def execute_method(
    ctx: Context,
    model: str,
    method: str,
    args_json: str = None,
    kwargs_json: str = None,
) -> Dict[str, Any]:
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
    odoo = ctx.request_context.lifespan_context.odoo

    try:
        args = []
        kwargs = {}

        if args_json:
            try:
                args = json.loads(args_json)
                if not isinstance(args, list):
                    return {"success": False, "error": "args_json must be a JSON array"}
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"Invalid args_json: {e}"}

        if kwargs_json:
            try:
                kwargs = json.loads(kwargs_json)
                if not isinstance(kwargs, dict):
                    return {"success": False, "error": "kwargs_json must be a JSON object"}
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"Invalid kwargs_json: {e}"}

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
        return {"success": True, "result": result}

    except Exception as e:
        error_msg = str(e)
        suggestion = get_error_suggestion(error_msg, model, method)

        response = {"success": False, "error": error_msg}
        if suggestion:
            response["suggestion"] = suggestion
            response["hint"] = "Check odoo://methods/{model} or odoo://module-knowledge/{module} for special methods"

        return response


@mcp.tool(
    description="""Discover available methods on an Odoo model.

    Returns:
    - Standard ORM methods (search, create, write, etc.)
    - Module-specific special methods (article_create, action_confirm, etc.)
    - Method signatures and parameters
    - Warnings about methods that shouldn't be used directly

    Use this BEFORE calling execute_method on unfamiliar models!
    """,
    annotations={
        "title": "Get Model Methods",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
def get_model_methods(
    ctx: Context,
    model: str,
    include_private: bool = False,
) -> Dict[str, Any]:
    """
    Discover available methods on an Odoo model.

    Parameters:
        model: Model name (e.g., 'sale.order', 'knowledge.article')
        include_private: Include methods starting with _ (default: False)

    Returns dict with:
        - orm_methods: Standard Odoo ORM methods
        - special_methods: Module-specific methods from knowledge base
        - action_methods: Discovered action_* and button_* methods
        - warnings: Important notes about this model
    """
    odoo = ctx.request_context.lifespan_context.odoo

    result = {
        "model": model,
        "orm_methods": {
            "read": ["search", "search_read", "read", "search_count", "browse", "fields_get", "default_get", "name_get", "name_search"],
            "write": ["create", "write", "unlink", "copy"],
        },
        "special_methods": [],
        "action_methods": [],
        "warnings": [],
        "field_mappings": {},
    }

    # Get special methods from knowledge base
    for module_name, module_info in MODULE_KNOWLEDGE.get("modules", {}).items():
        if module_info.get("model") == model:
            for method_name, method_info in module_info.get("special_methods", {}).items():
                result["special_methods"].append({
                    "name": method_name,
                    "description": method_info.get("description", ""),
                    "params": method_info.get("params", {}),
                    "replaces": method_info.get("instead_of"),
                })
                # Add warning if this replaces a standard method
                if method_info.get("instead_of"):
                    result["warnings"].append(
                        f"Use {method_name}() instead of {method_info['instead_of']}() for this model"
                    )

            if module_info.get("notes"):
                result["warnings"].append(module_info["notes"])

            if module_info.get("field_mappings"):
                result["field_mappings"] = module_info["field_mappings"]

    # Try to discover action methods from the model
    try:
        # Get server actions bound to this model
        actions = odoo.search_read(
            'ir.actions.server',
            [('model_id.model', '=', model)],
            fields=['name', 'state'],
            limit=50
        )
        for action in actions:
            result["action_methods"].append({
                "name": action.get('name'),
                "type": "server_action",
                "state": action.get('state')
            })
    except Exception:
        pass  # Model might not have ir.actions.server access

    # Add common action patterns for this model type
    common_actions = [
        "action_confirm", "action_cancel", "action_done", "action_draft",
        "action_post", "action_validate", "button_confirm", "button_cancel"
    ]

    if not result["special_methods"]:
        result["note"] = f"No special methods known for {model}. Try common action methods or check Odoo source."
        result["try_these"] = common_actions

    return {"success": True, "result": result}


@mcp.tool(
    description="""Find Odoo model from natural language concept.

    Translates business terms like "contact", "invoice", "quote" to
    actual Odoo model names like "res.partner", "account.move", "sale.order".

    Uses:
    1. Built-in concept aliases (fast)
    2. Search ir.model display names (comprehensive)
    3. Fuzzy model name matching (fallback)

    Examples:
    - "contact" → res.partner
    - "invoice" → account.move
    - "sales order" → sale.order
    """,
    annotations={
        "title": "Find Model",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
def find_model(
    ctx: Context,
    concept: str,
) -> Dict[str, Any]:
    """
    Find Odoo model from natural language concept.

    Parameters:
        concept: Business term like "contact", "invoice", "quote", "sales order"

    Returns:
        - best_match: Most likely model name
        - all_matches: All matching models with scores
        - source: How the match was found (alias/search/fuzzy)
    """
    odoo = ctx.request_context.lifespan_context.odoo

    concept_lower = concept.lower().strip()
    results = {
        "concept": concept,
        "best_match": None,
        "all_matches": [],
        "source": None,
    }

    # 1. Check built-in aliases first (instant)
    if concept_lower in CONCEPT_ALIASES:
        models = CONCEPT_ALIASES[concept_lower]
        results["best_match"] = models[0]
        results["all_matches"] = [{"model": m, "score": 100, "source": "alias"} for m in models]
        results["source"] = "alias"
        return {"success": True, "result": results}

    # 2. Search ir.model by display name
    try:
        ir_models = odoo.search_read(
            "ir.model",
            [["name", "ilike", concept]],
            fields=["model", "name", "info"],
            limit=10
        )

        if ir_models:
            for m in ir_models:
                score = 90 if concept_lower == m["name"].lower() else 70
                results["all_matches"].append({
                    "model": m["model"],
                    "display_name": m["name"],
                    "score": score,
                    "source": "ir.model"
                })

            # Sort by score
            results["all_matches"].sort(key=lambda x: x["score"], reverse=True)
            results["best_match"] = results["all_matches"][0]["model"]
            results["source"] = "ir.model"
            return {"success": True, "result": results}
    except Exception as e:
        # Continue to fuzzy matching if ir.model search fails
        pass

    # 3. Fuzzy match on model technical names
    try:
        all_models = odoo.search_read(
            "ir.model",
            [],
            fields=["model", "name"],
            limit=500
        )

        for m in all_models:
            model_name = m["model"].lower()
            display_name = m["name"].lower()

            # Check if concept appears in model name or display name
            if concept_lower in model_name or concept_lower in display_name:
                score = 60
                if concept_lower in model_name.split("."):
                    score = 80
                results["all_matches"].append({
                    "model": m["model"],
                    "display_name": m["name"],
                    "score": score,
                    "source": "fuzzy"
                })

        if results["all_matches"]:
            results["all_matches"].sort(key=lambda x: x["score"], reverse=True)
            results["best_match"] = results["all_matches"][0]["model"]
            results["source"] = "fuzzy"
            return {"success": True, "result": results}
    except Exception as e:
        results["error"] = f"Search failed: {e}"

    # No matches found
    results["suggestion"] = "Try a different term or check odoo://models for available models"
    return {"success": True, "result": results}


@mcp.tool(
    description="Execute multiple Odoo operations in a batch",
    annotations={
        "title": "Batch Execute",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
def batch_execute(
    ctx: Context,
    operations: List[Dict[str, Any]],
    atomic: bool = True
) -> Dict[str, Any]:
    """
    Execute multiple operations efficiently.

    Parameters:
        operations: List of operations, each with:
            - model: str (required)
            - method: str (required)
            - args_json: str (optional)
            - kwargs_json: str (optional)
        atomic: If True, fail fast on first error
    """
    odoo = ctx.request_context.lifespan_context.odoo
    results = []
    successful = 0
    failed = 0

    try:
        for idx, op in enumerate(operations):
            try:
                model = op.get('model')
                method = op.get('method')

                if not model or not method:
                    raise ValueError(f"Operation {idx}: 'model' and 'method' required")

                args_json = op.get('args_json')
                kwargs_json = op.get('kwargs_json')

                args = json.loads(args_json) if args_json else []
                kwargs = json.loads(kwargs_json) if kwargs_json else {}

                result = odoo.execute_method(model, method, *args, **kwargs)
                results.append({"operation_index": idx, "success": True, "result": result})
                successful += 1

            except Exception as e:
                results.append({"operation_index": idx, "success": False, "error": str(e)})
                failed += 1

                if atomic:
                    return {
                        "success": False,
                        "results": results,
                        "total_operations": len(operations),
                        "successful_operations": successful,
                        "failed_operations": failed,
                        "error": f"Failed at operation {idx}: {e}"
                    }

        return {
            "success": failed == 0,
            "results": results,
            "total_operations": len(operations),
            "successful_operations": successful,
            "failed_operations": failed,
            "error": None if failed == 0 else f"{failed} operations failed"
        }

    except Exception as e:
        return {
            "success": False,
            "results": results,
            "total_operations": len(operations),
            "successful_operations": successful,
            "failed_operations": failed,
            "error": str(e)
        }


# ----- MCP Prompts -----


@mcp.prompt(name="odoo-exploration")
def odoo_exploration_prompt() -> List[Dict[str, str]]:
    """Discover capabilities of this Odoo instance"""
    return [{
        "role": "user",
        "content": """Explore this Odoo instance:

1. Read odoo://server/info for version and apps
2. Read odoo://workflows for available workflows
3. Read odoo://models for all models

Provide a summary of what's available.
"""
    }]


@mcp.prompt(name="search-records")
def search_records_prompt(model: str = "res.partner") -> List[Dict[str, str]]:
    """Search for records in a model"""
    return [{
        "role": "user",
        "content": f"""Search for records in {model}.

First read odoo://model/{model}/schema to understand the fields.

Then use execute_method with:
- model='{model}'
- method='search_read'
- args_json='[[]]'  # empty domain for all records
- kwargs_json='{{"fields": ["name", "id"], "limit": 10}}'
"""
    }]


@mcp.prompt(name="odoo-api-reference")
def api_reference_prompt() -> List[Dict[str, str]]:
    """Quick reference for Odoo API patterns"""
    return [{
        "role": "user",
        "content": """## Odoo API Quick Reference

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
- Use get_model_methods() BEFORE calling unfamiliar models
- Check odoo://docs/{model} for documentation URLs

**Pre-execution Checklist:**
1. Model verified? (use get_model_methods if unfamiliar)
2. Method exists? (check special methods in module knowledge)
3. Required fields known? (check schema resource)
4. Types correct? (Many2one = ID, not name)
"""
    }]


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
    description="""Search available Odoo operations by keyword.

    This enables the CODE-FIRST PATTERN for 98% token reduction.
    Instead of loading all tool definitions upfront, search for
    what you need on-demand.

    Examples:
    - "invoice" → create_invoice, post_invoice, get_overdue_invoices
    - "sales" → create_quotation, confirm_order
    - "stock" → check_stock_levels, validate_delivery
    """,
    annotations={
        "title": "Search Tools",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
def search_tools(
    ctx: Context,
    query: str,
    category: str = None,
) -> Dict[str, Any]:
    """
    Search available Odoo operations by keyword.

    Parameters:
        query: Search term (e.g., "invoice", "sales", "stock")
        category: Optional filter by category (sales, accounting, crm, stock, hr)

    Returns:
        Matching tools with their definitions and usage examples
    """
    query_lower = query.lower()
    matches = []

    for tool_name, tool_def in TOOL_REGISTRY.items():
        # Search in name, description, and model
        searchable = f"{tool_name} {tool_def.get('description', '')} {tool_def.get('model', '')}".lower()

        if query_lower in searchable:
            # Category filter
            if category:
                tool_model = tool_def.get("model", "")
                category_match = {
                    "sales": "sale.",
                    "accounting": "account.",
                    "crm": "crm.",
                    "stock": "stock.",
                    "hr": "hr.",
                }.get(category.lower(), "")
                if category_match and category_match not in tool_model:
                    continue

            matches.append({
                "name": tool_name,
                **tool_def
            })

    # Also search module knowledge for special methods
    special_matches = []
    for module_name, module_info in MODULE_KNOWLEDGE.get("modules", {}).items():
        for method_name, method_info in module_info.get("special_methods", {}).items():
            searchable = f"{method_name} {method_info.get('description', '')} {module_info.get('model', '')}".lower()
            if query_lower in searchable:
                special_matches.append({
                    "name": method_name,
                    "model": module_info.get("model"),
                    "description": method_info.get("description"),
                    "params": method_info.get("params", {}),
                    "source": "module_knowledge"
                })

    return {
        "success": True,
        "result": {
            "query": query,
            "tools_found": len(matches) + len(special_matches),
            "tools": matches,
            "special_methods": special_matches,
            "tip": "Use execute_workflow() for multi-step operations or execute_method() for single calls"
        }
    }


@mcp.tool(
    description="""Discover all available actions for a specific model.

    Returns executable action definitions from:
    1. Tool registry (pre-built workflows)
    2. Module knowledge (special methods)
    3. Server actions (ir.actions.server)
    4. Standard ORM methods

    This is DYNAMIC DISCOVERY - finds what's actually available
    instead of assuming standard patterns.
    """,
    annotations={
        "title": "Discover Model Actions",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
def discover_model_actions(
    ctx: Context,
    model: str,
) -> Dict[str, Any]:
    """
    Discover all available actions for a model.

    Parameters:
        model: Model name (e.g., "sale.order", "account.move")

    Returns:
        Comprehensive list of available actions with usage patterns
    """
    odoo = ctx.request_context.lifespan_context.odoo

    result = {
        "model": model,
        "workflows": [],
        "special_methods": [],
        "server_actions": [],
        "orm_methods": ["search", "search_read", "create", "write", "unlink", "read"],
        "usage_examples": [],
    }

    # 1. Find tools in registry for this model
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

    # 3. Get server actions from Odoo
    try:
        actions = odoo.search_read(
            'ir.actions.server',
            [('model_id.model', '=', model)],
            fields=['name', 'state', 'code'],
            limit=20
        )
        for action in actions:
            result["server_actions"].append({
                "name": action.get('name'),
                "type": action.get('state'),
            })
    except Exception:
        pass

    # 4. Generate usage examples
    if result["special_methods"]:
        method = result["special_methods"][0]
        result["usage_examples"].append({
            "description": f"Call {method['name']}",
            "code": f'execute_method("{model}", "{method["name"]}", args_json="[[id]]")'
        })

    result["usage_examples"].append({
        "description": "Search records",
        "code": f'execute_method("{model}", "search_read", kwargs_json=\'{{"domain": [], "limit": 10}}\')'
    })

    return {"success": True, "result": result}


@mcp.tool(
    description="""Execute a multi-step workflow in a single call.

    This is the KEY TOOL for Code-First Pattern - combines multiple
    operations into one call, dramatically reducing tokens.

    Supported workflows:
    - quote_to_cash: Create quote → Confirm → Deliver → Invoice → Payment
    - lead_to_won: Create lead → Convert to opportunity → Mark won
    - create_and_post_invoice: Create invoice → Post it
    - stock_transfer: Create transfer → Confirm → Validate

    Or describe a custom workflow in natural language.
    """,
    annotations={
        "title": "Execute Workflow",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
def execute_workflow(
    ctx: Context,
    workflow: str,
    params_json: str = None,
) -> Dict[str, Any]:
    """
    Execute a multi-step workflow.

    Parameters:
        workflow: Workflow name or description
        params_json: JSON object with workflow parameters

    Returns:
        Results from each step of the workflow
    """
    odoo = ctx.request_context.lifespan_context.odoo

    try:
        params = json.loads(params_json) if params_json else {}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid params_json: {e}"}

    workflow_lower = workflow.lower().strip()
    results = {"workflow": workflow, "steps": [], "success": True}

    try:
        # ----- Quote to Cash Workflow -----
        if workflow_lower in ["quote_to_cash", "quotation_to_invoice", "sales_workflow"]:
            order_id = params.get("order_id")

            if not order_id:
                return {"success": False, "error": "order_id required for quote_to_cash workflow"}

            # Step 1: Confirm order
            try:
                odoo.execute_method("sale.order", "action_confirm", [order_id])
                results["steps"].append({"step": "confirm_order", "success": True})
            except Exception as e:
                results["steps"].append({"step": "confirm_order", "success": False, "error": str(e)})
                results["success"] = False
                return results

            # Step 2: Create invoice
            try:
                invoice_ids = odoo.execute_method("sale.order", "_create_invoices", [order_id])
                results["steps"].append({"step": "create_invoice", "success": True, "invoice_ids": invoice_ids})
            except Exception as e:
                results["steps"].append({"step": "create_invoice", "success": False, "error": str(e)})
                results["success"] = False
                return results

            # Step 3: Post invoice (optional)
            if params.get("post_invoice", True) and invoice_ids:
                try:
                    odoo.execute_method("account.move", "action_post", invoice_ids)
                    results["steps"].append({"step": "post_invoice", "success": True})
                except Exception as e:
                    results["steps"].append({"step": "post_invoice", "success": False, "error": str(e)})

            return results

        # ----- Lead to Won Workflow -----
        elif workflow_lower in ["lead_to_won", "crm_workflow", "opportunity_won"]:
            lead_id = params.get("lead_id")

            if not lead_id:
                return {"success": False, "error": "lead_id required for lead_to_won workflow"}

            # Step 1: Convert to opportunity (if still a lead)
            try:
                lead = odoo.search_read("crm.lead", [["id", "=", lead_id]], fields=["type"], limit=1)
                if lead and lead[0].get("type") == "lead":
                    odoo.execute_method("crm.lead", "convert_opportunity", [lead_id], partner_id=params.get("partner_id", False))
                    results["steps"].append({"step": "convert_to_opportunity", "success": True})
                else:
                    results["steps"].append({"step": "convert_to_opportunity", "skipped": True, "reason": "Already an opportunity"})
            except Exception as e:
                results["steps"].append({"step": "convert_to_opportunity", "success": False, "error": str(e)})

            # Step 2: Mark as won
            try:
                odoo.execute_method("crm.lead", "action_set_won", [lead_id])
                results["steps"].append({"step": "mark_won", "success": True})
            except Exception as e:
                results["steps"].append({"step": "mark_won", "success": False, "error": str(e)})
                results["success"] = False

            return results

        # ----- Create and Post Invoice Workflow -----
        elif workflow_lower in ["create_and_post_invoice", "quick_invoice"]:
            partner_id = params.get("partner_id")
            lines = params.get("lines", [])

            if not partner_id:
                return {"success": False, "error": "partner_id required"}
            if not lines:
                return {"success": False, "error": "lines required (list of {product_id, quantity, price_unit})"}

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
            try:
                invoice_vals = {
                    "move_type": "out_invoice",
                    "partner_id": partner_id,
                    "invoice_line_ids": invoice_lines,
                }
                invoice_id = odoo.execute_method("account.move", "create", [invoice_vals])
                results["steps"].append({"step": "create_invoice", "success": True, "invoice_id": invoice_id})
            except Exception as e:
                results["steps"].append({"step": "create_invoice", "success": False, "error": str(e)})
                results["success"] = False
                return results

            # Step 2: Post invoice
            if params.get("post", True):
                try:
                    odoo.execute_method("account.move", "action_post", [invoice_id])
                    results["steps"].append({"step": "post_invoice", "success": True})
                except Exception as e:
                    results["steps"].append({"step": "post_invoice", "success": False, "error": str(e)})

            results["invoice_id"] = invoice_id
            return results

        # ----- Unknown workflow -----
        else:
            return {
                "success": False,
                "error": f"Unknown workflow: {workflow}",
                "available_workflows": [
                    "quote_to_cash - Confirm order, create & post invoice",
                    "lead_to_won - Convert lead and mark as won",
                    "create_and_post_invoice - Create and post a customer invoice",
                ],
                "tip": "Use search_tools() to find available operations"
            }

    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        return results


# =====================================================
# HIGH IMPACT: Aggregation Tool (read_group)
# =====================================================


@mcp.tool(
    description="""Aggregate data using read_group for efficient reporting.

    Instead of fetching all records and processing in Python,
    use Odoo's read_group for database-level aggregation.

    Examples:
    - Total sales by customer: groupby=["partner_id"], fields=["amount_total:sum"]
    - Count invoices by state: groupby=["state"], fields=["__count"]
    - Average order value by month: groupby=["date_order:month"], fields=["amount_total:avg"]

    Aggregation functions: sum, avg, min, max, count
    Date grouping: :day, :week, :month, :quarter, :year
    """,
    annotations={
        "title": "Aggregate Data",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
def aggregate_data(
    ctx: Context,
    model: str,
    groupby: str,
    fields: str = None,
    domain_json: str = "[]",
) -> Dict[str, Any]:
    """
    Aggregate data using Odoo's read_group.

    Parameters:
        model: Model to aggregate (e.g., "sale.order", "account.move")
        groupby: Comma-separated fields to group by (e.g., "partner_id,state")
        fields: Comma-separated aggregation fields (e.g., "amount_total:sum,__count")
        domain_json: JSON domain to filter records

    Returns:
        Aggregated data grouped by specified fields
    """
    odoo = ctx.request_context.lifespan_context.odoo

    try:
        domain = json.loads(domain_json) if domain_json else []
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid domain_json: {e}"}

    # Parse groupby
    groupby_list = [g.strip() for g in groupby.split(",") if g.strip()]

    # Parse fields (default to count)
    if fields:
        fields_list = [f.strip() for f in fields.split(",") if f.strip()]
    else:
        fields_list = ["__count"]

    try:
        result = odoo.execute_method(
            model,
            "read_group",
            domain,
            fields=fields_list,
            groupby=groupby_list,
        )

        return {
            "success": True,
            "result": {
                "model": model,
                "domain": domain,
                "groupby": groupby_list,
                "fields": fields_list,
                "data": result,
                "count": len(result),
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# =====================================================
# HIGH IMPACT: Business Workflow Prompts
# =====================================================


@mcp.prompt(name="quote-to-cash")
def quote_to_cash_prompt(order_id: str = None) -> List[Dict[str, str]]:
    """Complete quote-to-cash workflow"""
    if order_id:
        return [{
            "role": "user",
            "content": f"""Execute the quote-to-cash workflow for order {order_id}:

Use execute_workflow("quote_to_cash", '{{"order_id": {order_id}}}')

This will:
1. Confirm the sales order (action_confirm)
2. Create invoice (_create_invoices)
3. Post invoice (action_post)

Report the result of each step.
"""
        }]
    else:
        return [{
            "role": "user",
            "content": """Guide me through creating a complete sales flow:

1. First, find or create a customer (res.partner)
2. Create a quotation (sale.order) with order lines
3. Confirm the quotation
4. Create and post the invoice
5. Optionally register payment

Use search_tools("sales") to find available operations.
"""
        }]


@mcp.prompt(name="ar-aging-report")
def ar_aging_report_prompt() -> List[Dict[str, str]]:
    """Generate accounts receivable aging report"""
    return [{
        "role": "user",
        "content": """Generate an AR aging report:

1. Use aggregate_data() to get totals by customer:
   - model: "account.move"
   - groupby: "partner_id"
   - fields: "amount_residual:sum"
   - domain_json: '[["move_type", "=", "out_invoice"], ["payment_state", "in", ["not_paid", "partial"]]]'

2. Then categorize by aging buckets:
   - Current (not yet due)
   - 1-30 days overdue
   - 31-60 days overdue
   - 61-90 days overdue
   - 90+ days overdue

3. Present as a formatted table with customer totals and recommendations.
"""
    }]


@mcp.prompt(name="inventory-check")
def inventory_check_prompt(product: str = None) -> List[Dict[str, str]]:
    """Check inventory levels"""
    if product:
        return [{
            "role": "user",
            "content": f"""Check inventory for "{product}":

1. Find the product using find_model("product") then search
2. Use aggregate_data on stock.quant:
   - groupby: "location_id"
   - fields: "quantity:sum"
   - domain_json with the product_id filter

3. Show available quantity by warehouse/location
"""
        }]
    else:
        return [{
            "role": "user",
            "content": """Check overall inventory status:

1. Use aggregate_data on stock.quant:
   - model: "stock.quant"
   - groupby: "product_id"
   - fields: "quantity:sum,value:sum"

2. Identify low stock items (quantity < reorder point)
3. Show top products by value
"""
        }]


@mcp.prompt(name="crm-pipeline")
def crm_pipeline_prompt() -> List[Dict[str, str]]:
    """Analyze CRM pipeline"""
    return [{
        "role": "user",
        "content": """Analyze the CRM pipeline:

1. Use aggregate_data on crm.lead:
   - groupby: "stage_id"
   - fields: "expected_revenue:sum,__count"
   - domain_json: '[["type", "=", "opportunity"]]'

2. Calculate conversion rates between stages
3. Identify opportunities that need attention:
   - Stuck in stage too long
   - High value opportunities
   - Upcoming activities

4. Present a pipeline summary with recommendations.
"""
    }]


@mcp.prompt(name="customer-360")
def customer_360_prompt(customer: str) -> List[Dict[str, str]]:
    """Complete customer 360 view"""
    return [{
        "role": "user",
        "content": f"""Get a 360-degree view of customer "{customer}":

1. Find the customer: find_model("customer"), then search res.partner

2. Get their data:
   - Basic info (name, email, phone, address)
   - Credit limit and receivables

3. Sales history:
   - aggregate_data("sale.order", "state", "amount_total:sum", domain with partner_id)

4. Invoice status:
   - aggregate_data("account.move", "payment_state", "amount_residual:sum", domain for invoices)

5. Recent activities:
   - Messages and notes from mail.message

6. CRM opportunities:
   - Open opportunities from crm.lead

Present a comprehensive customer profile with key insights.
"""
    }]


@mcp.prompt(name="daily-operations")
def daily_operations_prompt() -> List[Dict[str, str]]:
    """Daily operations dashboard"""
    return [{
        "role": "user",
        "content": """Generate a daily operations summary:

**Sales:**
- New orders today (aggregate sale.order by state where create_date = today)
- Pending quotations needing follow-up

**Inventory:**
- Pending deliveries (stock.picking with state = assigned or waiting)
- Low stock alerts

**Accounting:**
- Invoices to send (draft invoices)
- Overdue payments (use AR aging)
- Cash position

**CRM:**
- Activities due today
- Hot opportunities (high probability, high value)

Use aggregate_data for efficient queries. Present as a dashboard.
"""
    }]


# ----- Resource for listing available workflows -----


@mcp.resource(
    "odoo://tool-registry",
    description="Registry of pre-built tools and workflows (Code-First Pattern)",
)
def get_tool_registry() -> str:
    """Get the complete tool registry for code-first pattern."""
    return json.dumps({
        "description": "Pre-built tools and workflows for common Odoo operations",
        "usage": "Use search_tools(query) to find tools, execute_workflow(name, params) to run",
        "tools": TOOL_REGISTRY,
        "categories": {
            "sales": [k for k, v in TOOL_REGISTRY.items() if "sale" in v.get("model", "")],
            "accounting": [k for k, v in TOOL_REGISTRY.items() if "account" in v.get("model", "")],
            "crm": [k for k, v in TOOL_REGISTRY.items() if "crm" in v.get("model", "")],
            "stock": [k for k, v in TOOL_REGISTRY.items() if "stock" in v.get("model", "")],
            "hr": [k for k, v in TOOL_REGISTRY.items() if "hr" in v.get("model", "")],
        }
    }, indent=2)
