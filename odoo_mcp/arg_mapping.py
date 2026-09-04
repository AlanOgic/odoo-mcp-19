"""
Argument mapping for Odoo v2 JSON-2 API

Odoo 19+ uses JSON-2 API with named arguments only.
This module provides mapping from positional args to named args.

Every entry below is transcribed from the Odoo 19.0 source, not from the
docstrings (several of which are stale — ``default_get``'s still says
``fields_list`` while the parameter has been ``fields`` for releases):

  - ``odoo/orm/models.py`` @ 19.0 — search, search_read, search_count, read,
    name_search, name_create, copy, read_group, default_get, export_data,
    load, fields_get, write, create, unlink, has_access, check_access_rights,
    check_access_rule
  - ``addons/web/models/models.py`` @ 19.0 — formatted_read_group

A mapping MUST cover every positional parameter the method accepts. Any
positional argument without a name here is rejected by ``convert_args_to_v2``
rather than dropped — see the note on that function.
"""

from typing import Any

# Mapping of ORM method arguments from positional to named
# Format: method_name -> list of (arg_position, v2_param_name)
#
# Record-bound methods (read/write/unlink/copy/action_*) take their recordset
# in the JSON-2 body "ids" key, so position 0 is "ids" and the method's own
# signature starts at position 1.
V2_ARG_MAPPING: dict[str, list[tuple[int, str]]] = {
    # Search methods
    # search(domain, offset=0, limit=None, order=None)
    "search": [
        (0, "domain"),
        (1, "offset"),
        (2, "limit"),
        (3, "order"),
    ],
    # search_read(domain=None, fields=None, offset=0, limit=None, order=None, **read_kwargs)
    "search_read": [
        (0, "domain"),
        (1, "fields"),
        (2, "offset"),
        (3, "limit"),
        (4, "order"),
    ],
    # search_count(domain, limit=None)
    "search_count": [
        (0, "domain"),
        (1, "limit"),
    ],
    # Read methods
    # read(fields=None, load='_classic_read') — record-bound
    "read": [
        (0, "ids"),
        (1, "fields"),
        (2, "load"),
    ],
    # read_group(domain, fields, groupby, offset=0, limit=None, orderby=False, lazy=True)
    # Deprecated in v19 in favour of formatted_read_group, but still callable.
    # NOTE: this method's sort parameter is spelled "orderby", not "order".
    "read_group": [
        (0, "domain"),
        (1, "fields"),
        (2, "groupby"),
        (3, "offset"),
        (4, "limit"),
        (5, "orderby"),
        (6, "lazy"),
    ],
    # formatted_read_group (v19+ replacement for deprecated read_group)
    # formatted_read_group(domain, groupby=(), aggregates=(), having=(), offset=0, limit=None, order=None)
    "formatted_read_group": [
        (0, "domain"),
        (1, "groupby"),
        (2, "aggregates"),
        (3, "having"),
        (4, "offset"),
        (5, "limit"),
        (6, "order"),
    ],
    # Write methods
    # create(vals_list) — v2 API uses vals_list (array of dicts)
    "create": [
        (0, "vals_list"),
    ],
    # write(vals) — record-bound
    "write": [
        (0, "ids"),
        (1, "vals"),
    ],
    # unlink() — record-bound, takes nothing else
    "unlink": [
        (0, "ids"),
    ],
    # Name methods
    "name_get": [
        (0, "ids"),
    ],
    # name_search(name='', domain=None, operator='ilike', limit=100)
    "name_search": [
        (0, "name"),
        (1, "domain"),
        (2, "operator"),
        (3, "limit"),
    ],
    # name_create(name)
    "name_create": [
        (0, "name"),
    ],
    # Field methods
    # fields_get(allfields=None, attributes=None)
    "fields_get": [
        (0, "allfields"),
        (1, "attributes"),
    ],
    # default_get(fields) — the parameter is "fields"; the docstring's
    # "fields_list" is stale and sending it yields
    # "missing a required argument: 'fields'".
    "default_get": [
        (0, "fields"),
    ],
    # copy(default=None) — record-bound, so the selector belongs in "ids".
    "copy": [
        (0, "ids"),
        (1, "default"),
    ],
    # Check methods
    # check_access_rights(operation, raise_exception=True) — legacy wrapper, still present in 19.0
    "check_access_rights": [
        (0, "operation"),
        (1, "raise_exception"),
    ],
    # check_access_rule(operation) — legacy wrapper, still present in 19.0
    "check_access_rule": [
        (0, "operation"),
    ],
    # has_access(operation) — v19+ preferred form
    "has_access": [
        (0, "operation"),
    ],
    # Export/Import
    # export_data(fields_to_export)
    "export_data": [
        (0, "fields_to_export"),
    ],
    # load(fields, data)
    "load": [
        (0, "fields"),
        (1, "data"),
    ],
    # Action methods (common in Odoo). These run on a recordset, so the leading
    # [ids] argument must be forwarded as the JSON-2 body "ids" key — otherwise
    # the call executes against an empty recordset.
    "action_confirm": [(0, "ids")],
    "action_cancel": [(0, "ids")],
    "action_done": [(0, "ids")],
    "action_draft": [(0, "ids")],
    "action_validate": [(0, "ids")],
    "action_post": [(0, "ids")],
    # Workflow methods (same recordset semantics as the action methods above).
    "button_confirm": [(0, "ids")],
    "button_cancel": [(0, "ids")],
    "button_draft": [(0, "ids")],
    "button_validate": [(0, "ids")],
}


# Kwargs mapping: some kwargs have different names in v2
V2_KWARGS_MAPPING: dict[str, str] = {
    "fields": "fields",
    "offset": "offset",
    "limit": "limit",
    "order": "order",
    "context": "context",
    "attributes": "attributes",
    "lazy": "lazy",
    "orderby": "order",  # v1 uses orderby, v2 uses order
    "load": "load",  # read() param: '_classic_read' (default) or None for raw IDs
}

# read_group is the one survivor that really does spell its sort parameter
# "orderby" (odoo/orm/models.py @ 19.0). Renaming it to "order" there produces
# a 422, so the V2_KWARGS_MAPPING rename must not apply to it.
_KWARGS_MAPPING_EXEMPT: dict[str, set[str]] = {
    "read_group": {"orderby"},
}


def convert_args_to_v2(method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Convert positional arguments to named arguments for v2 API.

    Args:
        method: The ORM method name (e.g., 'search_read', 'write')
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Dictionary with all arguments as named parameters for v2 API

    Raises:
        ValueError: if a positional argument has no name to map to. JSON-2 is
            named-args-only, so an unmappable positional cannot be forwarded;
            dropping it silently would send a *different, still-valid* request
            (a name_search minus its domain, a search minus its limit) and
            return plausible but wrong results. Failing loudly is the point.
    """
    result: dict[str, Any] = {}

    # Get mapping for this method
    arg_mapping = V2_ARG_MAPPING.get(method, [])
    mapped_positions = {pos for pos, _ in arg_mapping}

    # Convert positional args
    consumed: set[int] = set()
    for pos, param_name in arg_mapping:
        if pos < len(args):
            result[param_name] = args[pos]
            consumed.add(pos)

    # Fallback for record-bound methods not in the table (e.g. action_set_won,
    # convert_opportunity): JSON-2 requires the recordset in the body "ids" key.
    # When position 0 isn't explicitly mapped and the leading arg looks like a
    # recordset (a list of record ids), route it to "ids" so it isn't dropped.
    if 0 not in mapped_positions and args:
        first = args[0]
        if isinstance(first, list) and all(isinstance(x, int) for x in first):
            result["ids"] = first
            consumed.add(0)

    # Refuse to silently drop a positional we have no name for.
    unmappable = [pos for pos in range(len(args)) if pos not in consumed]
    if unmappable:
        raise ValueError(_unmappable_args_message(method, args, arg_mapping, unmappable))

    # Convert kwargs (handle any name changes)
    exempt: set[str] = _KWARGS_MAPPING_EXEMPT.get(method, set())
    for k, v in kwargs.items():
        v2_name = k if k in exempt else V2_KWARGS_MAPPING.get(k, k)
        result[v2_name] = v

    # Ensure domain exists for search methods
    if method in ["search", "search_read", "search_count"]:
        if "domain" not in result:
            result["domain"] = []

    return result


def _unmappable_args_message(
    method: str,
    args: tuple[Any, ...],
    arg_mapping: list[tuple[int, str]],
    unmappable: list[int],
) -> str:
    """Build an actionable error for positional args with no JSON-2 name."""
    positions = ", ".join(str(p) for p in unmappable)
    plural = "s" if len(unmappable) > 1 else ""
    known = [name for _, name in sorted(arg_mapping)]

    parts = [
        f"{method}() was called with {len(args)} positional argument(s), but position{plural} "
        f"{positions} has no JSON-2 parameter name in V2_ARG_MAPPING. "
        "Odoo 19's JSON-2 API accepts named arguments only, so this value cannot be sent."
    ]
    if known:
        parts.append(f"Mapped positions for '{method}': {known}.")
    else:
        parts.append(f"'{method}' has no entry in V2_ARG_MAPPING.")
    parts.append(
        "Pass the extra argument(s) by name in kwargs_json instead "
        "(e.g. kwargs_json='{\"limit\": 5}'), or add the position to V2_ARG_MAPPING "
        "if this method should support it."
    )
    return " ".join(parts)


def get_supported_methods() -> list[str]:
    """Return list of methods with explicit v2 mapping support."""
    return list(V2_ARG_MAPPING.keys())


def is_method_supported(method: str) -> bool:
    """Check if a method has explicit v2 mapping support."""
    return method in V2_ARG_MAPPING
