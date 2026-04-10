"""
Pure data constants and validation functions for the Odoo MCP Server.

No local module imports except odoo_client (for module knowledge loading).
"""

import json
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


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

# ----- Default Context from env -----

_DEFAULT_CONTEXT: Optional[Dict[str, Any]] = None
_default_ctx_raw = os.environ.get("MCP_DEFAULT_CONTEXT")
if _default_ctx_raw:
    if len(_default_ctx_raw) > 4096:
        print("Warning: MCP_DEFAULT_CONTEXT exceeds 4KB, ignoring", file=sys.stderr)
    else:
        try:
            _DEFAULT_CONTEXT = json.loads(_default_ctx_raw)
            if not isinstance(_DEFAULT_CONTEXT, dict):
                print(f"Warning: MCP_DEFAULT_CONTEXT must be a JSON object, ignoring", file=sys.stderr)
                _DEFAULT_CONTEXT = None
        except json.JSONDecodeError as e:
            print(f"Warning: MCP_DEFAULT_CONTEXT invalid JSON: {e}", file=sys.stderr)


def _merge_context(explicit_context: Optional[Dict] = None) -> Optional[Dict]:
    """Merge MCP_DEFAULT_CONTEXT with explicit context. Explicit takes priority."""
    if not _DEFAULT_CONTEXT and not explicit_context:
        return None
    if not _DEFAULT_CONTEXT:
        return explicit_context
    if not explicit_context:
        return dict(_DEFAULT_CONTEXT)
    merged = dict(_DEFAULT_CONTEXT)
    merged.update(explicit_context)
    return merged


# ----- Input Validation -----

_MODEL_RE = re.compile(r'^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$')
_METHOD_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def _validate_model(model: str) -> Optional[str]:
    """Validate model name format. Returns error message or None if valid."""
    if not model or len(model) > 128:
        return "Model name is required and must be <= 128 characters"
    if not _MODEL_RE.match(model):
        return f"Invalid model name format: '{model}'. Expected dotted notation (e.g. 'res.partner')"
    return None


def _validate_method(method: str) -> Optional[str]:
    """Validate method name format. Returns error message or None if valid."""
    if not method or len(method) > 64:
        return "Method name is required and must be <= 64 characters"
    if not _METHOD_RE.match(method):
        return f"Invalid method name format: '{method}'. Expected identifier (e.g. 'search_read')"
    return None


# ----- Smart Limits -----

DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
ODOO_VERSION = "19.0"

# Known @api.private methods that cannot be called via RPC in Odoo 19
PRIVATE_METHOD_HINTS = {
    "check_access": "check_access is @api.private in v19. Use has_access(operation) instead (returns boolean).",
    "_read_group": "_read_group is @api.private. Use formatted_read_group (v19+) or read_group (deprecated but still works).",
    "search_fetch": "search_fetch is @api.private. Use search_read instead.",
    "fetch": "fetch is @api.private. Use read instead.",
}


# ----- State Machine Definitions -----


MODEL_STATE_MACHINES: Dict[str, Dict[str, Any]] = {
    "sale.order": {
        "state_field": "state",
        "states": ["draft", "sent", "sale", "done", "cancel"],
        "transitions": [
            {"from": "draft", "to": "sent", "method": "action_quotation_sent", "label": "Mark as Sent"},
            {"from": "draft", "to": "sale", "method": "action_confirm", "label": "Confirm Order",
             "side_effects": ["Creates delivery orders (stock.picking)", "Reserves stock"],
             "irreversible": False},
            {"from": "sent", "to": "sale", "method": "action_confirm", "label": "Confirm Order",
             "side_effects": ["Creates delivery orders (stock.picking)", "Reserves stock"],
             "irreversible": False},
            {"from": "sale", "to": "done", "method": "action_lock", "label": "Lock Order",
             "irreversible": False},
            {"from": ["draft", "sent", "sale"], "to": "cancel", "method": "action_cancel", "label": "Cancel",
             "irreversible": False},
        ],
    },
    "account.move": {
        "state_field": "state",
        "states": ["draft", "posted", "cancel"],
        "transitions": [
            {"from": "draft", "to": "posted", "method": "action_post", "label": "Post/Validate",
             "side_effects": ["Creates journal entries", "Updates account balances", "Assigns sequence number"],
             "irreversible": True},
            {"from": "posted", "to": "draft", "method": "button_draft", "label": "Reset to Draft",
             "side_effects": ["Removes sequence assignment"],
             "irreversible": False},
            {"from": "posted", "to": "cancel", "method": "button_cancel", "label": "Cancel",
             "irreversible": False},
        ],
    },
    "crm.lead": {
        "state_field": "type",
        "note": "CRM uses type (lead/opportunity) + stage_id, not a simple state field",
        "stages": "Dynamic - read crm.stage for available stages",
        "transitions": [
            {"from": "lead", "to": "opportunity", "method": "convert_opportunity", "label": "Convert to Opportunity",
             "side_effects": ["May create/link partner"],
             "irreversible": False},
            {"from": "opportunity", "to": "won", "method": "action_set_won", "label": "Mark Won",
             "side_effects": ["Updates probability to 100%"],
             "irreversible": False},
            {"from": "opportunity", "to": "lost", "method": "action_set_lost", "label": "Mark Lost",
             "side_effects": ["Archives the lead"],
             "irreversible": False},
        ],
    },
    "stock.picking": {
        "state_field": "state",
        "states": ["draft", "waiting", "confirmed", "assigned", "done", "cancel"],
        "transitions": [
            {"from": "draft", "to": "confirmed", "method": "action_confirm", "label": "Confirm",
             "irreversible": False},
            {"from": ["confirmed", "waiting"], "to": "assigned", "method": "action_assign", "label": "Check Availability",
             "side_effects": ["Reserves stock quantities"],
             "irreversible": False},
            {"from": "assigned", "to": "done", "method": "button_validate", "label": "Validate",
             "side_effects": ["Updates stock levels", "Creates stock moves"],
             "irreversible": True},
            {"from": ["draft", "confirmed", "assigned"], "to": "cancel", "method": "action_cancel", "label": "Cancel",
             "irreversible": False},
        ],
    },
    "purchase.order": {
        "state_field": "state",
        "states": ["draft", "sent", "purchase", "done", "cancel"],
        "transitions": [
            {"from": "draft", "to": "sent", "method": "action_rfq_send", "label": "Send RFQ",
             "irreversible": False},
            {"from": ["draft", "sent"], "to": "purchase", "method": "button_confirm", "label": "Confirm Order",
             "side_effects": ["Creates incoming receipt (stock.picking)"],
             "irreversible": False},
            {"from": "purchase", "to": "done", "method": "button_lock", "label": "Lock",
             "irreversible": False},
            {"from": ["draft", "sent", "purchase"], "to": "cancel", "method": "button_cancel", "label": "Cancel",
             "irreversible": False},
        ],
    },
    "hr.leave": {
        "state_field": "state",
        "states": ["draft", "confirm", "validate1", "validate", "refuse"],
        "transitions": [
            {"from": "draft", "to": "confirm", "method": "action_confirm", "label": "Confirm",
             "irreversible": False},
            {"from": "confirm", "to": "validate", "method": "action_approve", "label": "Approve",
             "side_effects": ["Deducts leave allocation"],
             "irreversible": False},
            {"from": ["confirm", "validate"], "to": "refuse", "method": "action_refuse", "label": "Refuse",
             "irreversible": False},
        ],
    },
}


# Runtime tracking of models that triggered fallback mechanisms
# Structure: {"model.name": {"method": {error_category: {...}}}}
RUNTIME_MODEL_ISSUES: Dict[str, Dict[str, Dict[str, Any]]] = {}
_RUNTIME_ISSUES_LOCK = threading.Lock()

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
_DOC_CACHE_MAX_ENTRIES = 100
_DOC_CACHE_LOCK = threading.Lock()


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


# ----- Bootstrap Models -----

_DEFAULT_BOOTSTRAP_MODELS = "res.partner,sale.order,account.move,product.product,stock.picking"


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


# ----- read_resource max chars -----

_READ_RESOURCE_MAX_CHARS = 15000  # Safe default for Claude Desktop context window
