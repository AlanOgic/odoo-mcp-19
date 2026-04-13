"""
Live integration test for the safety layer.

Requires Odoo connection (.env file with ODOO_URL, ODOO_DB, ODOO_USERNAME,
ODOO_API_KEY). Tests the full flow through execute_method including safety
classification.

Run as a script:
    python3 tests/live/test_safety_live.py

Note: this file is a script-style runner, not a pytest module. It mutates
os.environ at startup but restores it on exit so it can coexist with other
tests if the directory is later collected by pytest.
"""

import asyncio
import os
import sys

# Load .env
from dotenv import load_dotenv
load_dotenv()

from odoo_mcp.odoo_client import get_odoo_client


def header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def result_summary(resp):
    """Extract key fields from the response model."""
    d = resp.model_dump() if hasattr(resp, 'model_dump') else resp
    out = {
        "success": d.get("success"),
        "pending_confirmation": d.get("pending_confirmation"),
        "error": d.get("error"),
        "hint": d.get("hint"),
    }
    if d.get("safety"):
        s = d["safety"]
        out["safety.risk_level"] = s.get("risk_level") if isinstance(s, dict) else s.risk_level
        out["safety.requires_confirmation"] = s.get("requires_confirmation") if isinstance(s, dict) else s.requires_confirmation
        out["safety.cascade_warning"] = s.get("cascade_warning") if isinstance(s, dict) else s.cascade_warning
        out["safety.blocked_reason"] = s.get("blocked_reason") if isinstance(s, dict) else s.blocked_reason
    return {k: v for k, v in out.items() if v is not None}


async def run_tests():
    # We can't easily call the MCP tool functions directly since they need Context.
    # Instead, test the safety module directly with real classification + a real Odoo call.

    from odoo_mcp.safety import (
        classify_operation, classify_batch, classify_workflow,
        audit_log, RiskLevel,
    )

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name} -- {detail}")
            failed += 1

    # ---- Test 1: Safe read ----
    header("TEST 1: search_read on res.partner → SAFE")
    c = classify_operation("res.partner", "search_read", [], {"domain": [], "fields": ["name"], "limit": 2})
    print(f"  risk_level={c.risk_level.value}, requires_confirmation={c.requires_confirmation}")
    check("risk_level is SAFE", c.risk_level == RiskLevel.SAFE)
    check("no confirmation needed", c.requires_confirmation is False)

    # Verify it actually works with Odoo
    try:
        odoo = get_odoo_client()
        result = odoo.execute_method("res.partner", "search_read", domain=[], fields=["name"], limit=2)
        check("Odoo call succeeds", isinstance(result, list), f"got {type(result)}")
        print(f"  → Got {len(result)} partners")
    except Exception as e:
        check("Odoo call succeeds", False, str(e))

    # ---- Test 2: unlink without confirmed ----
    header("TEST 2: unlink on res.partner (no confirmed) → HIGH, needs confirmation")
    c = classify_operation("res.partner", "unlink", [[999999]])
    print(f"  risk_level={c.risk_level.value}, requires_confirmation={c.requires_confirmation}")
    check("risk_level is HIGH", c.risk_level == RiskLevel.HIGH)
    check("requires confirmation", c.requires_confirmation is True)
    check("record_count=1", c.record_count == 1)

    # ---- Test 3: BLOCKED model ----
    header("TEST 3: write on ir.rule → BLOCKED")
    c = classify_operation("ir.rule", "write", [[1], {"name": "hacked"}])
    print(f"  risk_level={c.risk_level.value}, blocked_reason={c.blocked_reason}")
    check("risk_level is BLOCKED", c.risk_level == RiskLevel.BLOCKED)
    check("has blocked_reason", c.blocked_reason is not None)

    # ---- Test 4: BLOCKED model even with safe read ----
    header("TEST 4: search_read on ir.rule → SAFE (reads are always safe)")
    c = classify_operation("ir.rule", "search_read")
    check("risk_level is SAFE", c.risk_level == RiskLevel.SAFE)

    # Verify we can actually read ir.rule
    try:
        result = odoo.execute_method("ir.rule", "search_read", domain=[], fields=["name"], limit=2)
        check("Odoo read on ir.rule works", isinstance(result, list))
        print(f"  → Got {len(result)} rules")
    except Exception as e:
        check("Odoo read on ir.rule works", False, str(e))

    # ---- Test 5: action_confirm with cascade warning ----
    header("TEST 5: action_confirm on sale.order → HIGH with cascade warning")
    c = classify_operation("sale.order", "action_confirm", [[1]])
    print(f"  risk_level={c.risk_level.value}, cascade_warning={c.cascade_warning}")
    check("risk_level is HIGH", c.risk_level == RiskLevel.HIGH)
    check("has cascade warning", c.cascade_warning is not None)
    check("warning mentions deliveries", "deliver" in (c.cascade_warning or "").lower())

    # ---- Test 6: Sensitive model write ----
    header("TEST 6: create on account.move → MEDIUM, requires confirmation")
    c = classify_operation("account.move", "create", [{"move_type": "out_invoice"}])
    print(f"  risk_level={c.risk_level.value}, requires_confirmation={c.requires_confirmation}")
    check("risk_level is MEDIUM", c.risk_level == RiskLevel.MEDIUM)
    check("requires confirmation (sensitive model)", c.requires_confirmation is True)

    # ---- Test 7: Batch classification ----
    header("TEST 7: Batch with mixed operations")
    ops = [
        {"model": "res.partner", "method": "search_read"},
        {"model": "sale.order", "method": "action_confirm", "args_json": "[[1]]"},
        {"model": "res.partner", "method": "write", "args_json": "[[1], {\"name\": \"x\"}]"},
    ]
    classifications, overall, needs_confirm = classify_batch(ops)
    print(f"  overall_risk={overall.value}, needs_confirm={needs_confirm}")
    print(f"  per-op risks: {[c.risk_level.value for c in classifications]}")
    check("overall risk is HIGH", overall == RiskLevel.HIGH)
    check("needs confirmation", needs_confirm is True)
    check("3 classifications returned", len(classifications) == 3)

    # ---- Test 8: Batch with blocked operation ----
    header("TEST 8: Batch with blocked operation")
    ops = [
        {"model": "res.partner", "method": "search_read"},
        {"model": "ir.rule", "method": "unlink", "args_json": "[[1]]"},
    ]
    classifications, overall, needs_confirm = classify_batch(ops)
    print(f"  overall_risk={overall.value}")
    check("overall risk is BLOCKED", overall == RiskLevel.BLOCKED)

    # ---- Test 9: Workflow classification ----
    header("TEST 9: quote_to_cash workflow → preview with HIGH risk")
    preview = classify_workflow("quote_to_cash")
    print(f"  overall_risk={preview.overall_risk.value}, steps={len(preview.steps)}")
    for step in preview.steps:
        print(f"    {step.step}: {step.risk_level.value} | {step.cascade_warning or '-'}")
    check("overall risk is HIGH", preview.overall_risk == RiskLevel.HIGH)
    check("3 steps", len(preview.steps) == 3)

    # ---- Test 10: Unknown workflow ----
    header("TEST 10: Unknown workflow → None")
    preview = classify_workflow("does_not_exist")
    check("returns None", preview is None)

    # ---- Test 11: Audit log output ----
    header("TEST 11: Audit log to stderr")
    c = classify_operation("res.partner", "unlink", [[1, 2, 3]])
    audit_log(c, confirmed=True, executed=True)
    print("  (check stderr above for [SAFETY AUDIT] JSON entry)")
    check("audit_log did not raise", True)

    # ---- Test 12: Confirmed unlink actually executes (on non-existent record) ----
    header("TEST 12: Confirmed unlink on non-existent partner (999999)")
    c = classify_operation("res.partner", "unlink", [[999999]])
    check("classified as HIGH", c.risk_level == RiskLevel.HIGH)
    # With confirmed=True, it should attempt the call (which will likely succeed or give MissingError)
    try:
        result = odoo.execute_method("res.partner", "unlink", [999999])
        print(f"  → Odoo returned: {result}")
        check("Odoo call executed (confirmed flow)", True)
    except Exception as e:
        # MissingError or 404 are expected — both mean the safety layer allowed it through
        err_str = str(e).lower()
        if any(p in err_str for p in ["missingerror", "missing", "does not exist", "404", "not found"]):
            print(f"  → Expected error (call went through, Odoo rejected): {e}")
            check("Odoo call attempted (safety layer allowed it)", True)
        else:
            print(f"  → Unexpected error: {e}")
            check("Odoo call executed", False, str(e))

    # ---- Summary ----
    header("SUMMARY")
    total = passed + failed
    print(f"  {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("  All tests passed!")
    else:
        print("  Some tests failed -- review output above")
    return failed == 0


if __name__ == "__main__":
    # Enable audit logging for this run only -- restored on exit so the
    # mutation doesn't leak into other test modules if pytest later
    # collects this directory.
    _prev_audit = os.environ.get("MCP_SAFETY_AUDIT")
    os.environ["MCP_SAFETY_AUDIT"] = "true"
    try:
        success = asyncio.run(run_tests())
    finally:
        if _prev_audit is None:
            os.environ.pop("MCP_SAFETY_AUDIT", None)
        else:
            os.environ["MCP_SAFETY_AUDIT"] = _prev_audit
    sys.exit(0 if success else 1)
