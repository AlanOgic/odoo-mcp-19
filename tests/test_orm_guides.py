"""Unit tests for the v1.18 ORM guide resources (points 4–8 and 10 of the API-reference plan).

Static guides live in ``odoo_mcp.orm_guides`` and are transcribed from the Odoo 19
source / docs; ``odoo://session`` is live (context_get + user + languages).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import odoo_mcp.app  # noqa: F401
import odoo_mcp.resources as resources
from odoo_mcp import orm_guides
from odoo_mcp.server import read_resource

# ----- odoo://api/datetime -----


def test_datetime_reference_pins_server_formats_and_utc_storage():
    ref = orm_guides.datetime_reference()
    assert ref["write_formats"]["date"] == "YYYY-MM-DD"
    assert ref["write_formats"]["datetime"] == "YYYY-MM-DD HH:MM:SS"
    assert "utc" in ref["storage"].lower() and "client" in ref["timezone"].lower()
    assert "T" in ref["rejected_forms"][0] or "iso" in ref["rejected_forms"][0].lower()


def test_datetime_reference_documents_dynamic_domain_values():
    ref = orm_guides.datetime_reference()
    assert ref["dynamic_domain_values"]["units"] == {
        "d": "days",
        "w": "weeks",
        "m": "months",
        "y": "years",
        "H": "hours",
        "M": "minutes",
        "S": "seconds",
    }
    assert "=monday -1w" in json.dumps(ref["dynamic_domain_values"]["examples"])


# ----- odoo://api/mail-thread -----


def test_mail_thread_reference_gives_the_keyword_only_message_post_signature():
    ref = orm_guides.mail_thread_reference()
    post = ref["message_post"]
    assert post["keyword_only"] is True
    assert "body_is_html" in post["params"] and "partner_ids" in post["params"]
    assert post["returns"].startswith("list") and "[123]" in post["returns"]  # JSON-2 unwraps the recordset to .ids
    assert "outgoing_email_to" in post["signature"]
    assert "ids" in post["json2_example"]


def test_mail_thread_reference_lists_the_notification_kill_switches():
    ref = orm_guides.mail_thread_reference()
    switches = ref["context_kill_switches"]
    for key in (
        "mail_notrack",
        "tracking_disable",
        "mail_create_nosubscribe",
        "mail_create_nolog",
        "mail_notify_force_send",
    ):
        assert key in switches


def test_mail_thread_reference_covers_followers_and_activities():
    ref = orm_guides.mail_thread_reference()
    assert "partner_ids" in ref["followers"]["message_subscribe"]
    assert "act_type_xmlid" in ref["activities"]["activity_schedule"]
    assert "activity_feedback" in ref["activities"]


# ----- odoo://api/security-model -----


def test_security_model_reference_states_the_composition_rules():
    ref = orm_guides.security_model_reference()
    assert "union" in ref["access_rights"]["composition"].lower()
    assert "allow" in ref["record_rules"]["default"].lower()
    assert "and" in ref["record_rules"]["global_rules"].lower() and "or" in ref["record_rules"]["group_rules"].lower()
    assert "fields_get" in ref["field_groups"]["symptom"]


def test_security_model_reference_points_to_the_callable_checks():
    ref = orm_guides.security_model_reference()
    assert "has_access" in ref["how_to_check"]
    assert "403" in ref["how_to_check"]["check_access"]
    assert "has_group" in ref["how_to_check"]


# ----- odoo://api/web-read -----


def test_web_read_reference_shows_the_nested_specification():
    ref = orm_guides.web_read_reference()
    spec = ref["specification"]["example"]
    assert spec["partner_id"]["fields"]["name"] == {}
    assert ref["methods"]["web_search_read"]["returns"].startswith("{'length'")
    assert "load" in ref["specification"]["many2one_without_fields"]


def test_web_read_reference_marks_web_save_as_a_write():
    ref = orm_guides.web_read_reference()
    assert ref["methods"]["web_save"]["side_effect"] is True
    assert ref["methods"]["web_read"]["side_effect"] is False


# ----- odoo://api/xmlids -----


def test_xmlids_reference_gives_the_only_public_resolver():
    ref = orm_guides.xmlids_reference()
    assert "check_object_reference" in ref["resolve"]["method"]
    assert ref["resolve"]["model"] == "ir.model.data"
    assert "env.ref" in ref["resolve"]["note"]


def test_xmlids_reference_lists_custom_model_constraints():
    ref = orm_guides.xmlids_reference()
    assert "x_" in ref["custom_models"]["name_prefix"]
    assert "manual" in ref["custom_models"]["state"]
    assert any("compute" in c.lower() for c in ref["custom_fields"]["not_possible"])


# ----- odoo://domain-syntax additions -----


def test_domain_syntax_now_documents_dynamic_dates_date_parts_and_any_bang():
    out = json.loads(resources.get_domain_syntax())
    assert out["dynamic_dates"]["units"]["H"] == "hours"
    assert "month_number" in out["date_parts"]["granularities"]
    assert "any!" in out["operators"]["relational"]


# ----- odoo://session (live) -----

_USER = {"id": 7, "name": "Bot", "login": "bot", "company_id": [1, "Cyanview"], "company_ids": [1, 2]}


def _session_client(context=None, user=_USER, langs=None, companies=None):
    client = MagicMock()

    def execute(model, method, *args, **kwargs):
        if (model, method) == ("res.users", "context_get"):
            return context if context is not None else {"lang": "en_US", "tz": "Europe/Brussels", "uid": 7}
        if (model, method) == ("res.lang", "get_installed"):
            return langs if langs is not None else [["en_US", "English (US)"], ["fr_FR", "French"]]
        raise AssertionError(f"unexpected call {model}.{method}")

    client.execute_method.side_effect = execute
    client.read_records.return_value = [user] if user else []
    client.search_read.return_value = (
        companies if companies is not None else [{"id": 1, "name": "Cyanview"}, {"id": 2, "name": "Cyanview US"}]
    )
    return client


def test_session_resource_assembles_identity_timezone_and_companies():
    client = _session_client()
    with patch.object(resources, "get_odoo_client", return_value=client):
        out = json.loads(resources.get_session())
    assert out["uid"] == 7 and out["login"] == "bot"
    assert out["lang"] == "en_US" and out["tz"] == "Europe/Brussels"
    assert out["company"] == {"id": 1, "name": "Cyanview"}
    assert out["allowed_companies"] == [{"id": 1, "name": "Cyanview"}, {"id": 2, "name": "Cyanview US"}]
    assert out["installed_languages"] == [
        {"code": "en_US", "name": "English (US)"},
        {"code": "fr_FR", "name": "French"},
    ]
    assert "allowed_company_ids" in out["context_hint"]


def test_session_resource_reads_the_user_by_the_uid_from_context_get():
    client = _session_client()
    with patch.object(resources, "get_odoo_client", return_value=client):
        resources.get_session()
    assert client.read_records.call_args.args[:2] == ("res.users", [7])
    assert client.search_read.call_args.kwargs.get("domain") == [["id", "in", [1, 2]]]


def test_session_resource_reports_a_failed_context_get_without_raising():
    client = MagicMock()
    client.execute_method.side_effect = ValueError("Request failed: 401 UNAUTHORIZED")
    with patch.object(resources, "get_odoo_client", return_value=client):
        out = json.loads(resources.get_session())
    assert "401" in out["error"]


# ----- bridge -----


@pytest.mark.parametrize(
    "uri",
    [
        "odoo://api/datetime",
        "odoo://api/mail-thread",
        "odoo://api/security-model",
        "odoo://api/web-read",
        "odoo://api/xmlids",
    ],
)
def test_bridge_routes_the_static_guides(uri):
    out = json.loads(read_resource(uri, max_chars=0))
    assert "error" not in out and out["source"]


def test_bridge_routes_session():
    with patch.object(resources, "get_odoo_client", return_value=_session_client()):
        out = json.loads(read_resource("odoo://session", max_chars=0))
    assert out["uid"] == 7
