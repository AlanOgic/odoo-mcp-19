"""
MCP Server for Odoo 19+

Provides MCP tools and resources for interacting with Odoo ERP via JSON-2 API.
"""

import json
import os
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


@mcp.resource(
    "odoo://module-knowledge",
    description="Module-specific methods and patterns knowledge base",
)
def get_module_knowledge() -> str:
    """Get the full module knowledge base with special methods and error patterns"""
    return json.dumps(MODULE_KNOWLEDGE, indent=2)


@mcp.resource(
    "odoo://module-knowledge/{module_name}",
    description="Get knowledge for a specific module (sale, crm, account, etc.)",
)
def get_module_knowledge_by_name(module_name: str) -> str:
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

    Common methods:
    - create: Create records
    - search_read: Search and read
    - write: Update records
    - unlink: Delete records
    - Custom methods: action_confirm, action_post, etc.

    Smart limits:
    - Default: 100 records
    - Maximum: 1000 records
    - Override with "limit" in kwargs_json
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
