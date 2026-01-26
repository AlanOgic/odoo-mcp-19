"""
Argument mapping for Odoo v2 JSON-2 API

Odoo 19+ uses JSON-2 API with named arguments only.
This module provides mapping from positional args to named args.
"""

from typing import Any, Dict, List, Tuple


# Mapping of ORM method arguments from positional to named
# Format: method_name -> list of (arg_position, v2_param_name)
V2_ARG_MAPPING: Dict[str, List[Tuple[int, str]]] = {
    # Search methods
    "search": [
        (0, "domain"),
    ],
    "search_read": [
        (0, "domain"),
    ],
    "search_count": [
        (0, "domain"),
    ],

    # Read methods
    "read": [
        (0, "ids"),
    ],
    "read_group": [
        (0, "domain"),
        (1, "fields"),
        (2, "groupby"),
    ],

    # Write methods
    "create": [
        (0, "vals"),
    ],
    "write": [
        (0, "ids"),
        (1, "vals"),
    ],
    "unlink": [
        (0, "ids"),
    ],

    # Name methods
    "name_get": [
        (0, "ids"),
    ],
    "name_search": [
        (0, "name"),
    ],
    "name_create": [
        (0, "name"),
    ],

    # Field methods
    "fields_get": [],
    "default_get": [
        (0, "fields_list"),
    ],

    # Copy/duplicate
    "copy": [
        (0, "id"),
    ],

    # Check methods
    "check_access_rights": [
        (0, "operation"),
    ],
    "check_access_rule": [
        (0, "operation"),
    ],

    # Export/Import
    "export_data": [
        (0, "fields_to_export"),
    ],
    "load": [
        (0, "fields"),
        (1, "data"),
    ],

    # Action methods (common in Odoo)
    "action_confirm": [],
    "action_cancel": [],
    "action_done": [],
    "action_draft": [],
    "action_validate": [],
    "action_post": [],

    # Workflow methods
    "button_confirm": [],
    "button_cancel": [],
    "button_draft": [],
    "button_validate": [],
}


# Kwargs mapping: some kwargs have different names in v2
V2_KWARGS_MAPPING: Dict[str, str] = {
    "fields": "fields",
    "offset": "offset",
    "limit": "limit",
    "order": "order",
    "context": "context",
    "attributes": "attributes",
    "lazy": "lazy",
    "orderby": "order",  # v1 uses orderby, v2 uses order
}


def convert_args_to_v2(
    method: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert positional arguments to named arguments for v2 API.

    Args:
        method: The ORM method name (e.g., 'search_read', 'write')
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Dictionary with all arguments as named parameters for v2 API
    """
    result: Dict[str, Any] = {}

    # Get mapping for this method
    arg_mapping = V2_ARG_MAPPING.get(method, [])

    # Convert positional args
    for pos, param_name in arg_mapping:
        if pos < len(args):
            result[param_name] = args[pos]

    # Convert kwargs (handle any name changes)
    for k, v in kwargs.items():
        v2_name = V2_KWARGS_MAPPING.get(k, k)
        result[v2_name] = v

    # Ensure domain exists for search methods
    if method in ['search', 'search_read', 'search_count']:
        if 'domain' not in result:
            result['domain'] = []

    return result


def get_supported_methods() -> List[str]:
    """Return list of methods with explicit v2 mapping support."""
    return list(V2_ARG_MAPPING.keys())


def is_method_supported(method: str) -> bool:
    """Check if a method has explicit v2 mapping support."""
    return method in V2_ARG_MAPPING
