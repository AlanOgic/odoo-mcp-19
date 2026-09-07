"""Static ORM guides behind ``odoo://api/{datetime,mail-thread,security-model,web-read,xmlids}``.

Each builder is pure and transcribed from a named Odoo 19 source or doc file
(see the ``source`` key of every payload) so a future refresh can diff against
the same place. Companion of ``api_reference.py``; split so neither module
grows past the project's file-size guideline.
"""

from __future__ import annotations

from typing import Any, Dict

_DATE_UNITS = {"d": "days", "w": "weeks", "m": "months", "y": "years", "H": "hours", "M": "minutes", "S": "seconds"}


def datetime_reference() -> Dict[str, Any]:
    """Server date formats, UTC storage, and the 19.0 dynamic domain values."""
    return {
        "write_formats": {"date": "YYYY-MM-DD", "datetime": "YYYY-MM-DD HH:MM:SS", "empty": "false"},
        "rejected_forms": [
            "ISO-8601 with 'T' separator or offset (2026-09-07T10:00:00+02:00) — not a server format",
            "milliseconds, Unix timestamps, locale strings",
        ],
        "storage": "Datetime columns are 'timestamp without time zone' holding UTC.",
        "timezone": "Timezone conversion is entirely client-side: JSON-2 returns naive UTC strings and expects "
        "naive UTC strings. Convert with the user's tz from res.users/context_get (see odoo://session).",
        "comparison": "Compare date fields to date strings and datetime fields to datetime strings only — "
        "a datetime string always sorts after a date string of the same day.",
        "dynamic_domain_values": {
            "since": "19.0",
            "rule": "In a domain, a date/datetime value may be a space-separated expression relative to now in the "
            "user's timezone: optional 'today' (midnight) or 'now', then terms starting with + (add), - (subtract) "
            "or = (set) followed by an integer and a unit, or a lower-case weekday.",
            "units": dict(_DATE_UNITS),
            "examples": {
                "now": ["date_order", "<", "now"],
                "today at midnight": ["date_order", ">=", "today"],
                "now minus 3 days plus 1 hour": ["write_date", ">=", "-3d +1H"],
                "today at 03:00": ["create_date", "<", "=3H"],
                "5th of this month": ["date", ">=", "=5d"],
                "Monday of last week": ["date", ">=", "=monday -1w"],
            },
        },
        "date_parts": {
            "since": "17.3",
            "rule": "A date field can be traversed to a numeric part in domains and groupby.",
            "granularities": [
                "year_number",
                "quarter_number",
                "month_number",
                "iso_week_number",
                "day_of_week",
                "day_of_month",
                "day_of_year",
                "hour_number",
                "minute_number",
                "second_number",
            ],
            "example": ["birthday.month_number", "=", 2],
        },
        "source": "developer/reference/backend/orm.rst (Date(time) Fields, Dynamic time values, Search domains)",
    }


def mail_thread_reference() -> Dict[str, Any]:
    """Chatter, followers and activities over JSON-2, with the notification kill-switches."""
    return {
        "message_post": {
            "keyword_only": True,
            "signature": "message_post(*, body='', subject=None, message_type='notification', email_from=None, "
            "author_id=None, parent_id=False, subtype_xmlid=None, subtype_id=False, partner_ids=None, "
            "outgoing_email_to=False, incoming_email_to=False, incoming_email_cc=False, attachments=None, "
            "attachment_ids=None, body_is_html=False, **kwargs)",
            "params": {
                "body": "Escaped as plain text unless body_is_html=true.",
                "body_is_html": "Set true to post HTML — documented 'to be used only for RPC calls'.",
                "message_type": "'comment' for a real note/message, 'notification' (default) for system lines.",
                "subtype_xmlid": "'mail.mt_comment' to send to followers, 'mail.mt_note' for an internal note.",
                "partner_ids": "List of res.partner ids to notify explicitly.",
                "attachment_ids": "Existing ir.attachment ids; 'attachments' takes [name, raw_content] pairs "
                "(not base64) and is awkward over JSON — create the ir.attachment first.",
            },
            "returns": "list with the created mail.message id, e.g. [123] — message_post returns a recordset and "
            "JSON-2 serialises any recordset as its list of ids",
            "json2_example": {
                "model": "sale.order",
                "method": "message_post",
                "ids": [15],
                "body": "Shipped today",
                "message_type": "comment",
                "subtype_xmlid": "mail.mt_note",
            },
            "access": "Posting requires write access on the record by default (_mail_post_access = 'write').",
        },
        "context_kill_switches": {
            "mail_notrack": "Skip field-change tracking messages for this write.",
            "tracking_disable": "Skip tracking and its notifications entirely.",
            "mail_create_nosubscribe": "Do not auto-subscribe the creator as follower on create.",
            "mail_create_nolog": "Skip the automatic 'created' chatter line.",
            "mail_notify_force_send": "Send e-mails immediately instead of queueing them.",
            "mail_post_autofollow": "Auto-follow recipients of a posted message.",
            "mail_auto_subscribe_no_notify": "Subscribe followers without notifying them.",
            "usage": "Pass them in the 'context' key of the JSON-2 body; a bulk write without mail_notrack "
            "e-mails every follower once per record.",
        },
        "followers": {
            "message_subscribe": "message_subscribe(partner_ids=None, subtype_ids=None) — ids in the body 'ids' key",
            "message_unsubscribe": "message_unsubscribe(partner_ids=None)",
        },
        "activities": {
            "activity_schedule": "activity_schedule(act_type_xmlid='', date_deadline=None, summary='', note='', "
            "**act_values) — e.g. act_type_xmlid='mail.mail_activity_data_todo', user_id=<res.users id>",
            "activity_feedback": "activity_feedback(act_type_xmlids, user_id=None, feedback=None) — marks done",
            "activity_unlink": "activity_unlink(act_type_xmlids, user_id=None) — removes without feedback",
            "model": "Activities live in mail.activity (res_model, res_id, activity_type_id, date_deadline, user_id).",
        },
        "source": "odoo/addons/mail/models/mail_thread.py, mail_activity_mixin.py; "
        "developer/reference/backend/mixins.rst",
    }


def security_model_reference() -> Dict[str, Any]:
    """How ACLs, record rules and field groups compose, and how to test access over JSON-2."""
    return {
        "access_rights": {
            "model": "ir.model.access",
            "semantics": "Grants read/write/create/unlink on a whole model to a group. No matching line = no access.",
            "composition": "Additive: a user's rights are the union of every group they belong to.",
            "perm_meaning": "perm_read / perm_write / perm_create / perm_unlink mean 'granted'.",
        },
        "record_rules": {
            "model": "ir.rule",
            "semantics": "Per-record conditions (domain_force) evaluated after the ACL, per operation.",
            "default": "Default-allow: if the ACL grants access and no rule applies, the record is accessible.",
            "global_rules": "Global rules (no groups) intersect — every one must pass (AND).",
            "group_rules": "Group rules unify — any one passing is enough (OR), within the bounds of the global rules.",
            "perm_meaning": "On ir.rule, perm_* means 'this rule applies to that operation' — "
            "the opposite of ir.model.access.",
            "domain_scope": "domain_force is a Python expression with time, user, company_id "
            "(int) and company_ids (list).",
            "multi_company": "Typical rule: ['|', ('company_id', '=', False), ('company_id', 'in', company_ids)] — "
            "so the allowed_company_ids context key changes what a search returns.",
        },
        "field_groups": {
            "semantics": "A field with groups='...' is invisible to users outside those groups.",
            "symptom": "The field is absent from fields_get and from views; explicit "
            "read/write of it raises AccessError.",
            "diagnosis": "A field present in the database schema but missing from fields_get means a field group, "
            "not a wrong model name.",
        },
        "how_to_check": {
            "has_access": "POST {model}/has_access with ids=[...] and "
            "operation='read'|'write'|'create'|'unlink' → bool. "
            "Empty ids = model-level check.",
            "check_access": "check_access(operation) raises instead of returning — but it is @api.private, so over "
            "JSON-2 it answers 403 itself; use has_access.",
            "has_group": "POST res.users/has_group with ids=[uid], group_ext_id='sales_team.group_sale_manager' → bool "
            "(only for the current user).",
            "error_shape": "A denied operation is odoo.exceptions.AccessError → HTTP 403; the message names the "
            "model and operation.",
        },
        "source": "developer/reference/backend/security.rst; odoo/orm/models.py (has_access, check_access)",
    }


def web_read_reference() -> Dict[str, Any]:
    """The nested ``specification`` used by web_read / web_search_read / web_save."""
    return {
        "why": "One call reads a record and its relations at any depth — replaces read + several follow-up reads.",
        "specification": {
            "rule": "A dict {field_name: spec}. Empty spec {} reads the raw value. For many2one, "
            "{'fields': {...}} expands the related record; for one2many/many2many, {'fields': {...}} reads the "
            "lines and accepts 'limit', 'order' and 'context'.",
            "example": {
                "name": {},
                "partner_id": {"fields": {"name": {}, "email": {}}},
                "order_line": {
                    "fields": {"product_id": {"fields": {"display_name": {}}}, "product_uom_qty": {}},
                    "limit": 50,
                },
            },
            "many2one_without_fields": "A many2one spec without 'fields' returns the bare id (the "
            "read uses load=None).",
            "display_name": "Ask for display_name explicitly inside 'fields' — it is not included by default.",
        },
        "methods": {
            "web_read": {
                "call": "POST {model}/web_read with ids=[...], specification={...}",
                "returns": "list of dicts, one per id",
                "side_effect": False,
            },
            "web_search_read": {
                "call": "POST {model}/web_search_read with domain, specification, offset, limit, order, count_limit",
                "returns": "{'length': <count>, 'records': [...]} — length is capped by count_limit if given",
                "side_effect": False,
            },
            "web_name_search": {
                "call": "POST {model}/web_name_search with name, specification, domain, operator, limit",
                "returns": "list of dicts shaped by specification (display_name included)",
                "side_effect": False,
            },
            "web_save": {
                "call": "POST {model}/web_save with ids=[] (create) or ids=[id] (write), "
                "vals={...}, specification={...}",
                "returns": "the saved record(s) re-read through specification",
                "side_effect": True,
            },
        },
        "gotcha": "web_* methods are what the Odoo web client uses; they honour access rules but skip nothing else — "
        "a web_save is a real write and goes through this server's safety gate.",
        "source": "odoo/addons/web/models/models.py (web_read, web_search_read, web_name_search, web_save)",
    }


def xmlids_reference() -> Dict[str, Any]:
    """External ids over JSON-2, and the constraints on runtime-created models and fields."""
    return {
        "what": "An XML id 'module.name' maps to (model, res_id) in ir.model.data. Views, groups, activity types, "
        "mail subtypes, report actions and sequences are addressed by XML id.",
        "resolve": {
            "model": "ir.model.data",
            "method": "check_object_reference(module, xml_id, raise_on_access_error=False) → [model, res_id]",
            "example": {
                "model": "ir.model.data",
                "method": "check_object_reference",
                "module": "mail",
                "xml_id": "mt_note",
            },
            "note": "env.ref() does not exist over JSON-2, and _xmlid_lookup / _xmlid_to_res_id are private (403). "
            "Alternatively search_read ir.model.data with domain [['module','=',m],['name','=',n]].",
            "reverse": "search_read ir.model.data with "
            "[['model','=',model],['res_id','=',id]] gives a record's XML ids.",
        },
        "noupdate": "ir.model.data.noupdate=true marks data the module update must not overwrite — user-edited "
        "records typically carry it.",
        "custom_models": {
            "model": "ir.model (BLOCKED for writes by this server)",
            "name_prefix": "Custom model names must start with x_",
            "state": "state must be 'manual' or the model is not loaded",
            "limits": "No methods can be added to a custom model, only fields.",
        },
        "custom_fields": {
            "model": "ir.model.fields (SENSITIVE: writes always confirm)",
            "required_keys": ["model_id", "name", "ttype", "field_description"],
            "optional_keys": [
                "required",
                "readonly",
                "translate",
                "groups",
                "selection",
                "size",
                "on_delete",
                "relation",
                "relation_field",
                "domain",
            ],
            "name_prefix": "Custom field names must start with x_ (Studio uses x_studio_)",
            "not_possible": [
                "Computed fields cannot be created via ir.model.fields",
                "Defaults and onchange logic cannot be attached",
                "Only state='manual' fields are activated",
            ],
        },
        "source": "odoo/addons/base/models/ir_model.py (check_object_reference); "
        "developer/reference/external_rpc_api.rst "
        "(ir.model, ir.model.fields); tutorials/define_module_data.rst",
    }
