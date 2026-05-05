"""Tests for MCP_WRITE_ALLOWLIST enforcement in classify_operation."""

from odoo_mcp.safety import RiskLevel, classify_operation


def test_safe_method_is_safe_even_under_locked(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "locked")
    result = classify_operation("res.partner", "search_read")
    assert result.risk_level is RiskLevel.SAFE


def test_side_effect_method_blocked_when_allowlist_enforced_and_empty(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "locked")
    result = classify_operation("sale.order", "action_confirm", args=[[1]])
    assert result.risk_level is RiskLevel.BLOCKED
    assert result.blocked_reason is not None
    assert "MCP_WRITE_ALLOWLIST" in result.blocked_reason
    assert "sale.order.action_confirm" in result.blocked_reason


def test_side_effect_method_allowed_when_in_allowlist(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "locked")
    monkeypatch.setenv("MCP_WRITE_ALLOWLIST", "sale.order.action_confirm")
    result = classify_operation("sale.order", "action_confirm", args=[[1]])
    assert result.risk_level is RiskLevel.HIGH


def test_wildcard_model_match(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "locked")
    monkeypatch.setenv("MCP_WRITE_ALLOWLIST", "sale.order.*")
    result = classify_operation("sale.order", "action_confirm", args=[[1]])
    assert result.risk_level is RiskLevel.HIGH


def test_wildcard_does_not_match_other_models(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "locked")
    monkeypatch.setenv("MCP_WRITE_ALLOWLIST", "sale.order.*")
    result = classify_operation("res.partner", "write", args=[[1], {"name": "X"}])
    assert result.risk_level is RiskLevel.BLOCKED


def test_explicit_allowlist_under_strict_enforces(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "strict")
    monkeypatch.setenv("MCP_WRITE_ALLOWLIST", "res.partner.message_post")
    result = classify_operation("res.partner", "write", args=[[1], {"name": "X"}])
    assert result.risk_level is RiskLevel.BLOCKED


def test_no_allowlist_no_enforcement(monkeypatch):
    """Without MCP_WRITE_ALLOWLIST set and not under locked, classifier
    behaviour is unchanged."""
    monkeypatch.setenv("MCP_SAFETY_MODE", "strict")
    monkeypatch.delenv("MCP_WRITE_ALLOWLIST", raising=False)
    result = classify_operation("res.partner", "write", args=[[1], {"name": "X"}])
    # write on res.partner under strict, single record → MEDIUM, no confirm.
    assert result.risk_level is RiskLevel.MEDIUM


def test_blocked_model_still_blocked_with_allowlist(monkeypatch):
    """The allowlist must NOT be able to override BLOCKED_MODELS."""
    monkeypatch.setenv("MCP_SAFETY_MODE", "locked")
    monkeypatch.setenv("MCP_WRITE_ALLOWLIST", "res.users.write")
    result = classify_operation("res.users", "write", args=[[1], {"name": "X"}])
    assert result.risk_level is RiskLevel.BLOCKED
    # The reason should still cite the security-critical model, not allowlist.
    assert ("security-critical" in (result.blocked_reason or "").lower()
            or "security-critical" in (result.reason or "").lower())
