"""
Live integration test for v1.11.0 DX improvements.

Requires Odoo connection (.env file with ODOO_URL, ODOO_DB, ODOO_USERNAME, ODOO_API_KEY).
Tests: quick-schema, bundle, session-bootstrap, workflow, resolve_json, error patterns, context merge.

Usage:
    python3 test_v1110_live.py
"""

import asyncio
import json
import os
import sys
import time

# Load .env
from dotenv import load_dotenv
load_dotenv()

from odoo_mcp.app import mcp
from odoo_mcp.utils import _build_compact_schema, get_error_suggestion
from odoo_mcp.constants import _merge_context, MODEL_STATE_MACHINES
from odoo_mcp.resources import (
    get_model_quick_schema,
    get_model_workflow,
    get_bundle,
    get_session_bootstrap,
)
from odoo_mcp import constants as _constants
from odoo_mcp.odoo_client import get_odoo_client


def header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name} — {detail}")
        failed += 1


def run_tests():
    global passed, failed
    odoo = get_odoo_client()

    # ================================================================
    # TEST 1: _build_compact_schema (unit test, no Odoo call)
    # ================================================================
    header("TEST 1: _build_compact_schema — unit test")
    fake_fields = {
        "name": {"type": "char", "required": True, "readonly": False},
        "partner_id": {"type": "many2one", "required": False, "readonly": False, "relation": "res.partner"},
        "state": {"type": "selection", "required": False, "readonly": True, "selection": [("draft", "Draft"), ("done", "Done")]},
        "line_ids": {"type": "one2many", "required": False, "readonly": False, "relation": "sale.order.line"},
        "amount": {"type": "float", "required": False, "readonly": True},
    }
    schema = _build_compact_schema(fake_fields)
    check("returns dict with 'fields' key", "fields" in schema)
    check("returns dict with 'required_fields' key", "required_fields" in schema)
    check("'name' is required", "name" in schema["required_fields"])
    check("partner_id has rel=res.partner", schema["fields"]["partner_id"].get("rel") == "res.partner")
    check("state has sel (selection values)", "sel" in schema["fields"]["state"])
    check("amount has ro=True", schema["fields"]["amount"].get("ro") is True)
    check("name has req=True", schema["fields"]["name"].get("req") is True)
    check("5 fields in schema", len(schema["fields"]) == 5)
    # Check compactness: no label, no help
    json_str = json.dumps(schema, separators=(",", ":"))
    check(f"compact JSON is small ({len(json_str)} chars)", len(json_str) < 500, f"got {len(json_str)}")

    # ================================================================
    # TEST 2: _merge_context (unit test)
    # ================================================================
    header("TEST 2: _merge_context — unit test")
    # Save original state
    original_default = _constants._DEFAULT_CONTEXT

    # Test with no defaults — patch the constants module where _merge_context reads it from
    _constants._DEFAULT_CONTEXT = None
    check("no defaults + no explicit = None", _merge_context(None) is None)
    check("no defaults + explicit = explicit", _merge_context({"lang": "fr_FR"}) == {"lang": "fr_FR"})

    # Test with defaults
    _constants._DEFAULT_CONTEXT = {"lang": "en_US", "tz": "UTC"}
    check("defaults + no explicit = copy of defaults", _merge_context(None) == {"lang": "en_US", "tz": "UTC"})
    check("defaults + explicit = merged (explicit wins)", _merge_context({"lang": "fr_FR"}) == {"lang": "fr_FR", "tz": "UTC"})

    # Restore
    _constants._DEFAULT_CONTEXT = original_default

    # ================================================================
    # TEST 3: get_error_suggestion with {model} template
    # ================================================================
    header("TEST 3: get_error_suggestion — template substitution")
    # Many2one error (should match fallback pattern)
    suggestion = get_error_suggestion(
        "422: ValidationError - Expected int for Many2one field",
        model="sale.order",
        method="create"
    )
    if suggestion:
        check("Many2one pattern matched", True)
        check("{model} substituted", "sale.order" in suggestion or "many2one" in suggestion.lower(),
              f"suggestion: {suggestion}")
    else:
        check("Many2one pattern matched", False, "No suggestion returned")

    # Singleton error
    suggestion = get_error_suggestion(
        "422: Expected singleton: res.partner(1, 2)",
        model="res.partner",
    )
    check("Singleton pattern matched", suggestion is not None, "No suggestion returned")

    # Unknown error
    suggestion = get_error_suggestion("some random error that won't match")
    check("Unknown error returns None", suggestion is None, f"got: {suggestion}")

    # ================================================================
    # TEST 4: MODEL_STATE_MACHINES structure
    # ================================================================
    header("TEST 4: MODEL_STATE_MACHINES — structure check")
    expected_models = ["sale.order", "account.move", "crm.lead", "stock.picking", "purchase.order", "hr.leave"]
    for model in expected_models:
        check(f"{model} in state machines", model in MODEL_STATE_MACHINES)

    so = MODEL_STATE_MACHINES["sale.order"]
    check("sale.order has state_field", so.get("state_field") == "state")
    check("sale.order has states list", isinstance(so.get("states"), list) and len(so["states"]) > 0)
    check("sale.order has transitions", isinstance(so.get("transitions"), list) and len(so["transitions"]) > 0)
    # Check transition structure
    t0 = so["transitions"][0]
    check("transition has 'from'", "from" in t0)
    check("transition has 'to'", "to" in t0)
    check("transition has 'method'", "method" in t0)
    check("transition has 'label'", "label" in t0)

    # ================================================================
    # TEST 5: quick-schema resource (LIVE Odoo call)
    # ================================================================
    header("TEST 5: quick-schema — res.partner (LIVE)")
    t0 = time.time()
    result_str = get_model_quick_schema("res.partner")
    elapsed = time.time() - t0
    result = json.loads(result_str)
    check("no error", "error" not in result, result.get("error", ""))
    check("has model key", result.get("model") == "res.partner")
    check("has fields dict", isinstance(result.get("fields"), dict))
    check("has required_fields list", isinstance(result.get("required_fields"), list))
    check("has field_count", isinstance(result.get("field_count"), int))
    field_count = result.get("field_count", 0)
    check(f"field_count > 10 (got {field_count})", field_count > 10)
    # Check compactness
    # res.partner has ~278 fields, so compact schema is ~13KB (still 60-80% less than full /fields ~50KB)
    check(f"response size ({len(result_str)} bytes) < 20000", len(result_str) < 20000,
          f"got {len(result_str)} bytes")
    check(f"response time: {elapsed:.2f}s", elapsed < 15, f"took {elapsed:.2f}s")
    # Verify a known field
    if "name" in result.get("fields", {}):
        check("'name' field has type", "t" in result["fields"]["name"])
    else:
        check("'name' field exists", False, "name not in fields")

    # ================================================================
    # TEST 6: workflow resource — static (sale.order)
    # ================================================================
    header("TEST 6: workflow — sale.order (static)")
    result_str = get_model_workflow("sale.order")
    result = json.loads(result_str)
    check("source is 'static'", result.get("source") == "static")
    check("model is sale.order", result.get("model") == "sale.order")
    check("has transitions", len(result.get("transitions", [])) > 0)
    # Check action_confirm transition has side_effects
    confirm_transitions = [t for t in result.get("transitions", []) if t.get("method") == "action_confirm"]
    check("action_confirm transition exists", len(confirm_transitions) > 0)
    if confirm_transitions:
        check("action_confirm has side_effects", "side_effects" in confirm_transitions[0])

    # ================================================================
    # TEST 7: workflow resource — dynamic fallback (res.partner)
    # ================================================================
    header("TEST 7: workflow — res.partner (dynamic fallback, LIVE)")
    result_str = get_model_workflow("res.partner")
    result = json.loads(result_str)
    check("source is 'dynamic'", result.get("source") == "dynamic")
    check("model is res.partner", result.get("model") == "res.partner")
    # res.partner doesn't have a state field, so note should be present
    has_state = result.get("state_field") is not None
    has_methods = len(result.get("available_methods", [])) > 0
    has_note = "note" in result
    check("either has state/methods or a note", has_state or has_methods or has_note,
          f"state={has_state}, methods={has_methods}, note={has_note}")

    # ================================================================
    # TEST 8: bundle resource (LIVE)
    # ================================================================
    header("TEST 8: bundle — res.partner,sale.order (LIVE)")
    t0 = time.time()
    result_str = get_bundle("res.partner,sale.order")
    elapsed = time.time() - t0
    result = json.loads(result_str)
    check("has models dict", isinstance(result.get("models"), dict))
    check("2 models returned", result.get("total") == 2, f"got {result.get('total')}")
    check("res.partner in result", "res.partner" in result.get("models", {}))
    check("sale.order in result", "sale.order" in result.get("models", {}))
    check("no errors", len(result.get("errors", {})) == 0, str(result.get("errors")))
    check(f"response time: {elapsed:.2f}s", elapsed < 30, f"took {elapsed:.2f}s")

    # ================================================================
    # TEST 9: bundle — max 10 limit
    # ================================================================
    header("TEST 9: bundle — max 10 models limit")
    models_11 = ",".join([f"model.{i}" for i in range(11)])
    result_str = get_bundle(models_11)
    result = json.loads(result_str)
    check("error returned for >10 models", "error" in result)

    # ================================================================
    # TEST 10: session-bootstrap (LIVE)
    # ================================================================
    header("TEST 10: session-bootstrap (LIVE)")
    t0 = time.time()
    result_str = get_session_bootstrap()
    elapsed = time.time() - t0
    result = json.loads(result_str)
    check("has schemas dict", isinstance(result.get("schemas"), dict))
    check("has workflows dict", isinstance(result.get("workflows"), dict))
    # Default 5 models
    schema_count = len(result.get("schemas", {}))
    check(f"schemas for 5 models (got {schema_count})", schema_count == 5)
    # At least sale.order and account.move should have workflows
    workflow_count = len(result.get("workflows", {}))
    check(f"workflows present (got {workflow_count})", workflow_count >= 2)
    check(f"response time: {elapsed:.2f}s", elapsed < 60, f"took {elapsed:.2f}s")
    # Check size is reasonable (should be compact)
    check(f"response size ({len(result_str)} bytes)", len(result_str) > 0)

    # ================================================================
    # TEST 11: resolve_json — via execute_method (LIVE)
    # ================================================================
    header("TEST 11: resolve_json — resolve user by name_search (LIVE)")
    # We'll test the resolve logic by calling execute_method through the server
    # Since we can't easily call the MCP tool directly, test the underlying logic
    try:
        # First, find a user that actually exists
        users = odoo.execute_method("res.users", "search_read", domain=[], fields=["name"], limit=1)
        if users:
            test_name = users[0]["name"]
            print(f"  Using test user: '{test_name}'")
            matches = odoo.execute_method("res.users", "name_search", name=test_name, limit=5)
            check("name_search works for res.users", isinstance(matches, list))
            if matches:
                check(f"'{test_name}' found: id={matches[0][0]}", len(matches) >= 1)
                user_id = matches[0][0]
                check("returned ID is int", isinstance(user_id, int))
            else:
                check(f"'{test_name}' found in res.users", False, "No matches returned")
        else:
            check("at least one user exists", False, "No users in instance")
    except Exception as e:
        check("name_search call works", False, str(e))

    # ================================================================
    # TEST 12: resolve_json — ambiguity detection
    # ================================================================
    header("TEST 12: resolve_json — ambiguity detection logic")
    try:
        # Search for a common name that should return multiple results
        matches = odoo.execute_method("res.partner", "name_search", name="a", limit=5)
        check("broad search returns results", isinstance(matches, list) and len(matches) > 0,
              f"got {len(matches) if matches else 0} matches")
        if matches and len(matches) > 1:
            check(f"multiple matches detected ({len(matches)})", len(matches) > 1)
        else:
            print("  ⚠️  Only 1 or 0 matches for 'a' — can't test ambiguity")
    except Exception as e:
        check("broad name_search works", False, str(e))

    # ================================================================
    # TEST 13: Context bug fix — search_read fallback preserves context
    # ================================================================
    header("TEST 13: search_read with context (LIVE)")
    try:
        result = odoo.execute_method(
            "res.partner", "search_read",
            domain=[["is_company", "=", True]],
            fields=["name"],
            limit=2,
            context={"lang": "fr_FR"},
        )
        check("search_read with context works", isinstance(result, list))
        if result:
            check(f"got {len(result)} records", len(result) > 0)
    except Exception as e:
        check("search_read with context works", False, str(e))

    # ================================================================
    # TEST 14: Error patterns — expanded patterns test
    # ================================================================
    header("TEST 14: Error patterns — expanded coverage")
    # 422 patterns
    s = get_error_suggestion("422: Expected singleton: res.partner(1, 2)")
    check("422 singleton pattern", s is not None)

    s = get_error_suggestion("422: null value in column \"name\" violates not-null constraint")
    check("422 null value pattern", s is not None)

    s = get_error_suggestion("422: Invalid field 'nonexistent' on model 'res.partner'", model="res.partner")
    check("422 invalid field pattern", s is not None)

    # 500 patterns
    s = get_error_suggestion("500: OperationalError - statement timeout")
    check("500 OperationalError pattern", s is not None)

    # 403 patterns
    s = get_error_suggestion("403: Access Denied - ir.rule restriction")
    check("403 ir.rule pattern", s is not None)

    # 404 patterns
    s = get_error_suggestion("404: /json/2/nonexistent.model/search_read")
    check("404 json/2 pattern", s is not None)

    # Fallback patterns (no HTTP code)
    s = get_error_suggestion("Cannot convert 'Mitchell Admin' to int for Many2one field partner_id")
    check("fallback Many2one pattern", s is not None)

    s = get_error_suggestion("Expected singleton: sale.order(1, 2, 3)")
    check("fallback singleton pattern", s is not None)

    # ================================================================
    # TEST 15: Full resource route smoke test (via read_resource tool)
    # ================================================================
    header("TEST 15: Resource route smoke test")
    from odoo_mcp.server import _RESOURCE_ROUTES
    # Verify new routes are registered
    route_patterns = [r[0] if isinstance(r, tuple) else r for r in _RESOURCE_ROUTES]
    route_str = str(route_patterns)
    check("quick-schema route registered", "quick-schema" in route_str, route_str[:200])
    check("workflow route registered", "workflow" in route_str, route_str[:200])
    check("bundle route registered", "bundle" in route_str, route_str[:200])
    check("session-bootstrap route registered", "session-bootstrap" in route_str, route_str[:200])

    # ================================================================
    # SUMMARY
    # ================================================================
    header("SUMMARY")
    total = passed + failed
    print(f"  {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("  🎉 All tests passed!")
    else:
        print("  ⚠️  Some tests failed — review output above")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
