"""Static API reference payloads and the compact ``/doc-bearer/index.json`` catalogue.

Everything here is transcribed from the Odoo 19 source and developer docs —
cite the file next to each block so a future refresh can diff against the
same place:

* ``odoo/addons/rpc/controllers/json2.py`` (the single ``/json/2`` route)
* ``odoo/http.py`` (``serialize_exception``, ``Json2Dispatcher.handle_error``)
* ``odoo/orm/commands.py`` (the ``Command`` enum docstring)
* ``content/developer/reference/backend/orm/changelog.rst``
* ``content/developer/reference/external_api.rst`` ("Transaction")

The three ``*_reference()`` builders are pure: no Odoo call, no env.
``build_api_index`` compacts the live index so a 2 MB payload becomes a
per-model summary an agent can scan in one read.
"""

from __future__ import annotations

from typing import Any, Dict, List

_PR = "https://github.com/odoo/odoo/pull/"


def json2_protocol_reference() -> Dict[str, Any]:
    """The ``/json/2`` request/response contract, as implemented in Odoo 19."""
    return {
        "endpoint": "POST {ODOO_URL}/json/2/{model}/{method}",
        "headers": {
            "Authorization": "Bearer <api key> (global-scope key; the 'rpc' scope maps to it)",
            "Content-Type": "application/json (anything else is 415)",
            "X-Odoo-Database": "optional — needed only when the host serves several databases",
        },
        "body": {
            "ids": "Recordset the method runs on. Omit (or send []) for @api.model methods "
            "such as search_read, fields_get, create, name_search — passing ids to one is a 422.",
            "context": "Dict merged into the environment context (lang, tz, allowed_company_ids, active_test…).",
            "<param>": "Every other key is bound by name to the Python signature.",
        },
        "positional_args": False,
        "positional_args_note": "The controller binds kwargs with inspect.signature(func).bind(records, **kwargs); "
        "there is no way to pass a positional argument. This server maps them via arg_mapping.",
        "private_methods": "Methods starting with '_' or decorated @api.private raise AccessError: 403 "
        "(not 404 — the method exists but is not exposed).",
        "return_value": "The bare JSON of the Python return value — no {'result': …} envelope. "
        "A returned recordset is serialised as its list of ids.",
        "error_body_keys": ["name", "message", "arguments", "context", "debug"],
        "error_body_note": "name is the fully qualified exception class (odoo.exceptions.AccessError, "
        "werkzeug.exceptions.NotFound…); arguments is exc.args; debug is the server traceback "
        "(this server logs it to stderr and never forwards it).",
        "status_codes": {
            "400": "Body is not valid JSON.",
            "401": "Missing or invalid bearer key (WWW-Authenticate: bearer).",
            "403": "odoo.exceptions.AccessError — ACL, record rule or field group denies the operation, "
            "or the method is private ('_' prefix / @api.private).",
            "404": "Unknown model or unknown method name, or odoo.exceptions.MissingError (record deleted).",
            "409": "odoo.exceptions.LockError — the record is locked by another transaction; retry later.",
            "415": "Content-Type is not application/json — the response is HTML, not JSON.",
            "422": "Signature bind failed (unknown or missing named parameter), ids sent to an @api.model "
            "method, or a business rule (odoo.exceptions.UserError / ValidationError) — read message.",
            "500": "Any other exception (ValueError, KeyError, psycopg2 errors…) — check message.",
        },
        "transactions": {
            "rule": "Every call runs in its own SQL transaction: committed on success, discarded on error. "
            "Calls cannot be chained inside one transaction.",
            "advice": "Prefer one composite method (search_read, action_confirm, action_post…) over a "
            "sequence of primitive calls: concurrent requests can change the data between two calls. "
            "batch_execute(atomic=True) stops at the first failure but does not roll back earlier ops.",
        },
        "useful_calls": {
            "current user": "POST res.users/context_get (no ids) → {lang, tz, uid} of the key's owner",
            "server version": "GET {ODOO_URL}/web/version → {version, version_info}",
            "raw many2one ids": "read with kwargs {'load': null} returns ints instead of [id, display_name]",
            "xml id lookup": "ir.model.data/check_object_reference(module, xml_id) → [model, res_id]; "
            "env.ref() does not exist over JSON-2",
        },
        "rate_limiting": {
            "odoo_core": None,
            "note": "Odoo itself defines no 429 on the HTTP RPC path (only a websocket limit). "
            "A 429 comes from the hosting infrastructure in front of Odoo, before execution.",
            "this_server": "Read methods are retried once after a 429 (Retry-After honoured, capped); "
            "write methods are never retried.",
        },
        "sources": [
            "odoo/addons/rpc/controllers/json2.py",
            "odoo/http.py (serialize_exception, Json2Dispatcher.handle_error)",
            "developer/reference/external_api.rst",
        ],
    }


def version_drift_reference() -> Dict[str, Any]:
    """ORM renames and removals since Odoo 15.2 that break calls written from older knowledge."""
    changes: List[Dict[str, str]] = [
        {
            "since": "19.1",
            "old": "ir.config_parameter get_param/set_param usage patterns",
            "use": "new ir.config_parameter API (writes on this model stay BLOCKED by this server)",
            "json2_impact": "Read parameters with search_read on ir.config_parameter; do not script set_param.",
            "pr": f"{_PR}223180",
        },
        {
            "since": "19.0",
            "old": "static date strings in domains",
            "use": "dynamic values: 'today', 'now', '-3d +1H', '=monday -1w' (units d w m y H M S)",
            "json2_impact": "Domains can carry relative dates; see odoo://domain-syntax.",
            "pr": f"{_PR}216665",
        },
        {
            "since": "18.2",
            "old": "read_group",
            "use": "formatted_read_group(domain, groupby, aggregates, …) — param is aggregates, not fields",
            "json2_impact": "read_group still answers but is deprecated; _read_group is private (404).",
            "pr": f"{_PR}163300",
        },
        {
            "since": "18.2",
            "old": "calling any public Python method over RPC",
            "use": "only methods without @api.private — check_access/search_fetch are private, "
            "use has_access/search_read",
            "json2_impact": "@api.private methods raise AccessError (403) even though they are public in Python.",
            "pr": f"{_PR}195402",
        },
        {
            "since": "18.0",
            "old": "check_access_rights",
            "use": "check_access(operation) (raises) or has_access(operation) (bool) — both combine ACL + record rules",
            "json2_impact": "Still callable in 19.0 but @api.deprecated (server logs a warning); has_access is the "
            "RPC-friendly replacement (check_access is @api.private → 403).",
            "pr": f"{_PR}179148",
        },
        {
            "since": "18.0",
            "old": "check_access_rule",
            "use": "has_access(operation) — record rules are now checked together with the ACL",
            "json2_impact": "Still callable in 19.0 but @api.deprecated; prefer has_access.",
            "pr": f"{_PR}179148",
        },
        {
            "since": "18.0",
            "old": "_name_search overrides / name_search semantics",
            "use": "_search_display_name — name_search still works and delegates to it",
            "json2_impact": "None for callers; explains why custom name matching follows display_name.",
            "pr": f"{_PR}174967",
        },
        {
            "since": "17.3",
            "old": "grouping by a date field only at day/week/month granularity",
            "use": "date-part groupby and domain traversal: 'date.month_number', 'date.day_of_week'…",
            "json2_impact": "formatted_read_group groupby accepts 'field:month_number'.",
            "pr": f"{_PR}159528",
        },
        {
            "since": "17.2",
            "old": "group_operator",
            "use": "aggregator",
            "json2_impact": "fields_get exposes 'aggregator'; a domain or aggregate on 'group_operator' fails.",
            "pr": f"{_PR}127353",
        },
        {
            "since": "16.4",
            "old": "name_get",
            "use": "read the display_name field",
            "json2_impact": "name_get is deprecated; read(fields=['display_name']) or search_read.",
            "pr": f"{_PR}122085",
        },
        {
            "since": "16.2",
            "old": "search + read",
            "use": "search_fetch / fetch exist but are @api.private — use search_read over JSON-2",
            "json2_impact": "search_fetch is @api.private: 403 over RPC despite appearing in older examples.",
            "pr": f"{_PR}112126",
        },
        {
            "since": "16.0",
            "old": "search_count ignoring limit",
            "use": "search_count(domain, limit=N) stops counting at N",
            "json2_impact": "Pass limit to bound the cost of an existence check.",
            "pr": f"{_PR}95589",
        },
        {
            "since": "15.3",
            "old": "args",
            "use": "domain — first parameter of search, search_count and _search",
            "json2_impact": "JSON-2 is named-args-only: {'args': [...]} is a 422, send {'domain': [...]}.",
            "pr": f"{_PR}83687",
        },
        {
            "since": "15.3",
            "old": "fields_get_keys, get_xml_id, browse('12')",
            "use": "list(fields_get()), ir.model.data lookup, browse(12)",
            "json2_impact": "Deprecated helpers; string ids are rejected.",
            "pr": f"{_PR}83687",
        },
    ]
    return {
        "description": "ORM API changes since Odoo 15.2 that make calls written from older docs fail over JSON-2",
        "usage": "When a method name or parameter 404s/422s, look it up here before retrying.",
        "changes": changes,
        "source": "developer/reference/backend/orm/changelog.rst",
    }


def x2many_commands_reference() -> Dict[str, Any]:
    """The literal One2many / Many2many command triples accepted over RPC."""
    return {
        "rpc_rule": "Over RPC only the literal 3-element triple [code, id, value] is accepted — "
        "neither Command.link(...) nor the constant names exist in JSON.",
        "commands": [
            {
                "code": 0,
                "name": "CREATE",
                "shape": "[0, 0, values]",
                "effect": "Create a record in the comodel with values and link it.",
                "notes": "On a many2many one shared record is created; on a one2many one per record in ids.",
            },
            {
                "code": 1,
                "name": "UPDATE",
                "shape": "[1, id, values]",
                "effect": "Write values on the linked record id.",
                "notes": "The record must already be linked.",
            },
            {
                "code": 2,
                "name": "DELETE",
                "shape": "[2, id, 0]",
                "effect": "Unlink then delete the record id from the database.",
                "notes": "Irreversible. Some many2many relations refuse it — use 3 to just detach.",
            },
            {
                "code": 3,
                "name": "UNLINK",
                "shape": "[3, id, 0]",
                "effect": "Remove the relation to id; the record is kept unless the inverse many2one cascades.",
                "notes": "On a one2many whose inverse has ondelete='cascade' (e.g. sale.order.line.order_id) "
                "UNLINK deletes the line exactly like DELETE; otherwise it only nulls the inverse field, "
                "which fails if that field is required.",
            },
            {
                "code": 4,
                "name": "LINK",
                "shape": "[4, id, 0]",
                "effect": "Add a relation to the existing record id.",
                "notes": "Idempotent on many2many.",
            },
            {
                "code": 5,
                "name": "CLEAR",
                "shape": "[5, 0, 0]",
                "effect": "Remove every relation (records are kept).",
                "notes": "",
            },
            {
                "code": 6,
                "name": "SET",
                "shape": "[6, 0, ids]",
                "effect": "Replace the whole relation with ids.",
                "notes": "Most common for many2many: tag_ids = [[6, 0, [1, 2, 3]]].",
            },
        ],
        "examples": {
            "add two tags": {"tag_ids": [[4, 7, 0], [4, 9, 0]]},
            "replace all tags": {"tag_ids": [[6, 0, [7, 9]]]},
            "create an order line": {"order_line": [[0, 0, {"product_id": 42, "product_uom_qty": 2}]]},
            "update a line": {"order_line": [[1, 315, {"product_uom_qty": 3}]]},
            "delete a line": {"order_line": [[2, 315, 0]]},
        },
        "source": "odoo/orm/commands.py (class Command docstring)",
    }


def build_api_index(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Compact ``/doc-bearer/index.json``: keep model names and counts, drop the field maps.

    The raw index lists every readable field's label for every model (~2 MB on
    a full instance); this keeps a ~80-byte line per model so the whole catalogue
    fits one read. Field detail belongs to ``odoo://model/{m}/quick-schema``.
    """
    modules = list(raw.get("modules") or [])
    models = sorted(
        (
            {
                "model": entry.get("model", ""),
                "name": entry.get("name", ""),
                "field_count": len(entry.get("fields") or {}),
                "method_count": len(entry.get("methods") or []),
            }
            for entry in raw.get("models") or []
        ),
        key=lambda m: m["model"],
    )
    return {
        "description": "Installed modules (dependency order) and every model readable by this API user",
        "usage": "Pick a model here, then read odoo://model/{model}/quick-schema and odoo://methods/{model}",
        "source": "/doc-bearer/index.json",
        "module_count": len(modules),
        "model_count": len(models),
        "modules": modules,
        "models": models,
    }
