"""Unit tests for the per-role safety gate (readonly profiles)."""

from odoo_mcp.safety import RiskLevel, classify_batch, classify_operation


def test_readonly_blocks_write():
    c = classify_operation("res.partner", "write", [[1], {"name": "x"}], role="readonly")
    assert c.risk_level == RiskLevel.BLOCKED
    assert "Read-only profile" in c.reason


def test_readonly_blocks_create():
    c = classify_operation("sale.order", "create", [{}], role="readonly")
    assert c.risk_level == RiskLevel.BLOCKED


def test_readonly_allows_safe_methods():
    c = classify_operation("res.partner", "search_read", role="readonly")
    assert c.risk_level == RiskLevel.SAFE


def test_other_roles_unaffected():
    for role in (None, "admin", "support"):
        c = classify_operation("res.partner", "search_read", role=role)
        assert c.risk_level == RiskLevel.SAFE


def test_batch_threads_role():
    ops = [
        {"model": "res.partner", "method": "search_read"},
        {"model": "res.partner", "method": "write", "args_json": "[[1], {}]"},
    ]
    classifications, overall, _ = classify_batch(ops, role="readonly")
    assert classifications[0].risk_level == RiskLevel.SAFE
    assert classifications[1].risk_level == RiskLevel.BLOCKED
    assert overall == RiskLevel.BLOCKED
