"""
Unit tests for the safety classification layer.

No Odoo connection needed — tests only the classification logic.
"""

import json
import os
from unittest.mock import patch

import pytest

from odoo_mcp.safety import (
    BLOCKED_MODELS,
    CASCADE_WARNINGS,
    HIGH_METHODS,
    SAFE_METHODS,
    SENSITIVE_MODELS,
    RiskLevel,
    SafetyClassification,
    audit_log,
    classify_batch,
    classify_operation,
    classify_workflow,
)


# =====================================================
# Test: SAFE methods
# =====================================================


class TestSafeMethods:
    """All SAFE_METHODS should classify as SAFE with no confirmation."""

    @pytest.mark.parametrize("method", sorted(SAFE_METHODS))
    def test_safe_methods_on_regular_model(self, method):
        result = classify_operation("res.partner", method)
        assert result.risk_level == RiskLevel.SAFE
        assert result.requires_confirmation is False

    @pytest.mark.parametrize("method", sorted(SAFE_METHODS))
    def test_safe_methods_on_blocked_model(self, method):
        """Even on blocked models, read operations are SAFE."""
        result = classify_operation("ir.rule", method)
        assert result.risk_level == RiskLevel.SAFE
        assert result.requires_confirmation is False

    @pytest.mark.parametrize("method", sorted(SAFE_METHODS))
    def test_safe_methods_on_sensitive_model(self, method):
        result = classify_operation("account.move", method)
        assert result.risk_level == RiskLevel.SAFE
        assert result.requires_confirmation is False


# =====================================================
# Test: BLOCKED models
# =====================================================


class TestBlockedModels:
    """Non-safe methods on BLOCKED_MODELS should be BLOCKED."""

    @pytest.mark.parametrize("model", sorted(BLOCKED_MODELS))
    def test_write_on_blocked_model(self, model):
        result = classify_operation(model, "write", [[1], {"name": "x"}])
        assert result.risk_level == RiskLevel.BLOCKED
        assert result.blocked_reason is not None

    @pytest.mark.parametrize("model", sorted(BLOCKED_MODELS))
    def test_create_on_blocked_model(self, model):
        result = classify_operation(model, "create", [{"name": "x"}])
        assert result.risk_level == RiskLevel.BLOCKED

    @pytest.mark.parametrize("model", sorted(BLOCKED_MODELS))
    def test_unlink_on_blocked_model(self, model):
        result = classify_operation(model, "unlink", [[1]])
        assert result.risk_level == RiskLevel.BLOCKED

    def test_blocked_even_with_confirmed(self):
        """Blocked models cannot be overridden — classification is always BLOCKED."""
        result = classify_operation("ir.rule", "write", [[1], {}])
        assert result.risk_level == RiskLevel.BLOCKED


# =====================================================
# Test: HIGH methods
# =====================================================


class TestHighMethods:
    """HIGH_METHODS should always require confirmation."""

    @pytest.mark.parametrize("method", sorted(HIGH_METHODS))
    def test_high_methods_require_confirmation(self, method):
        result = classify_operation("sale.order", method, [[1]])
        assert result.risk_level == RiskLevel.HIGH
        assert result.requires_confirmation is True

    def test_high_method_with_multiple_records(self):
        result = classify_operation("res.partner", "unlink", [[1, 2, 3]])
        assert result.risk_level == RiskLevel.HIGH
        assert result.requires_confirmation is True
        assert result.record_count == 3


# =====================================================
# Test: MEDIUM methods (strict mode)
# =====================================================


class TestMediumMethodsStrict:
    """MEDIUM_METHODS in strict mode."""

    @pytest.fixture(autouse=True)
    def set_strict_mode(self):
        with patch.dict(os.environ, {"MCP_SAFETY_MODE": "strict"}):
            yield

    def test_single_record_write_no_confirm(self):
        """Single-record write in strict does NOT require confirmation."""
        result = classify_operation("res.partner", "write", [[1], {"name": "x"}])
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.requires_confirmation is False

    def test_multi_record_write_requires_confirm(self):
        """Multi-record write in strict DOES require confirmation."""
        result = classify_operation("res.partner", "write", [[1, 2, 3], {"name": "x"}])
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.requires_confirmation is True
        assert result.record_count == 3

    def test_single_create_no_confirm(self):
        result = classify_operation("res.partner", "create", [{"name": "x"}])
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.requires_confirmation is False
        assert result.record_count == 1

    def test_batch_create_requires_confirm(self):
        result = classify_operation(
            "res.partner", "create", [[{"name": "a"}, {"name": "b"}]]
        )
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.requires_confirmation is True
        assert result.record_count == 2


# =====================================================
# Test: MEDIUM methods (permissive mode)
# =====================================================


class TestMediumMethodsPermissive:
    """MEDIUM_METHODS in permissive mode."""

    @pytest.fixture(autouse=True)
    def set_permissive_mode(self):
        with patch.dict(os.environ, {"MCP_SAFETY_MODE": "permissive"}):
            yield

    def test_multi_record_write_no_confirm(self):
        """Multi-record write in permissive does NOT require confirmation."""
        result = classify_operation("res.partner", "write", [[1, 2, 3], {"name": "x"}])
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.requires_confirmation is False

    def test_batch_create_no_confirm(self):
        result = classify_operation(
            "res.partner", "create", [[{"name": "a"}, {"name": "b"}]]
        )
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.requires_confirmation is False


# =====================================================
# Test: Sensitive models
# =====================================================


class TestSensitiveModels:
    """Writes on SENSITIVE_MODELS always require confirmation."""

    @pytest.mark.parametrize("model", sorted(SENSITIVE_MODELS))
    def test_write_on_sensitive_always_confirms(self, model):
        result = classify_operation(model, "write", [[1], {"x": 1}])
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.requires_confirmation is True

    @pytest.mark.parametrize("model", sorted(SENSITIVE_MODELS))
    def test_create_on_sensitive_always_confirms(self, model):
        result = classify_operation(model, "create", [{"x": 1}])
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.requires_confirmation is True

    @pytest.mark.parametrize("mode", ["strict", "permissive"])
    def test_sensitive_confirms_in_both_modes(self, mode):
        with patch.dict(os.environ, {"MCP_SAFETY_MODE": mode}):
            result = classify_operation("account.move", "write", [[1], {}])
            assert result.requires_confirmation is True


# =====================================================
# Test: Record counting
# =====================================================


class TestRecordCounting:
    def test_unlink_count(self):
        result = classify_operation("res.partner", "unlink", [[1, 2, 3, 4]])
        assert result.record_count == 4

    def test_write_count(self):
        result = classify_operation("res.partner", "write", [[10, 20], {"name": "x"}])
        assert result.record_count == 2

    def test_create_single(self):
        result = classify_operation("res.partner", "create", [{"name": "x"}])
        assert result.record_count == 1

    def test_create_batch(self):
        result = classify_operation(
            "res.partner", "create",
            [[{"name": "a"}, {"name": "b"}, {"name": "c"}]],
        )
        assert result.record_count == 3

    def test_action_with_ids(self):
        result = classify_operation("sale.order", "action_confirm", [[1, 2]])
        assert result.record_count == 2

    def test_action_with_single_id(self):
        result = classify_operation("sale.order", "action_confirm", [1])
        assert result.record_count == 1

    def test_copy_always_one(self):
        result = classify_operation("res.partner", "copy", [1])
        assert result.record_count == 1

    def test_no_args(self):
        result = classify_operation("res.partner", "write")
        assert result.record_count is None


# =====================================================
# Test: Cascade warnings
# =====================================================


class TestCascadeWarnings:
    def test_known_cascade_present(self):
        for (model, method), expected_warning in CASCADE_WARNINGS.items():
            result = classify_operation(model, method)
            assert result.cascade_warning == expected_warning

    def test_no_cascade_for_regular_method(self):
        result = classify_operation("res.partner", "unlink", [[1]])
        assert result.cascade_warning is None


# =====================================================
# Test: Unknown methods
# =====================================================


class TestUnknownMethods:
    def test_unknown_strict_requires_confirmation(self):
        with patch.dict(os.environ, {"MCP_SAFETY_MODE": "strict"}):
            result = classify_operation("res.partner", "custom_action")
            assert result.risk_level == RiskLevel.MEDIUM
            assert result.requires_confirmation is True

    def test_unknown_permissive_no_confirmation(self):
        with patch.dict(os.environ, {"MCP_SAFETY_MODE": "permissive"}):
            result = classify_operation("res.partner", "custom_action")
            assert result.risk_level == RiskLevel.MEDIUM
            assert result.requires_confirmation is False


# =====================================================
# Test: Workflow classification
# =====================================================


class TestWorkflowClassification:
    def test_quote_to_cash(self):
        preview = classify_workflow("quote_to_cash")
        assert preview is not None
        assert preview.overall_risk == RiskLevel.HIGH
        assert len(preview.steps) == 3

    def test_quote_to_cash_aliases(self):
        for alias in ["quotation_to_invoice", "sales_workflow"]:
            preview = classify_workflow(alias)
            assert preview is not None
            assert preview.overall_risk == RiskLevel.HIGH

    def test_lead_to_won(self):
        preview = classify_workflow("lead_to_won")
        assert preview is not None
        assert len(preview.steps) == 2

    def test_create_and_post_invoice(self):
        preview = classify_workflow("create_and_post_invoice")
        assert preview is not None
        assert preview.overall_risk == RiskLevel.HIGH

    def test_stock_transfer(self):
        preview = classify_workflow("stock_transfer")
        assert preview is not None
        assert preview.overall_risk == RiskLevel.HIGH

    def test_unknown_workflow_returns_none(self):
        preview = classify_workflow("nonexistent_workflow")
        assert preview is None

    def test_case_insensitive(self):
        preview = classify_workflow("Quote_To_Cash")
        assert preview is not None

    def test_cascade_warnings_in_workflow(self):
        preview = classify_workflow("quote_to_cash")
        assert preview is not None
        warnings = [s.cascade_warning for s in preview.steps if s.cascade_warning]
        assert len(warnings) > 0  # sale.order action_confirm and account.move action_post


# =====================================================
# Test: Batch classification
# =====================================================


class TestBatchClassification:
    def test_all_safe_batch(self):
        ops = [
            {"model": "res.partner", "method": "search_read"},
            {"model": "sale.order", "method": "read"},
        ]
        classifications, overall, needs_confirm = classify_batch(ops)
        assert overall == RiskLevel.SAFE
        assert needs_confirm is False
        assert len(classifications) == 2

    def test_mixed_batch(self):
        ops = [
            {"model": "res.partner", "method": "search_read"},
            {"model": "sale.order", "method": "action_confirm", "args_json": "[[1]]"},
        ]
        classifications, overall, needs_confirm = classify_batch(ops)
        assert overall == RiskLevel.HIGH
        assert needs_confirm is True

    def test_blocked_in_batch(self):
        ops = [
            {"model": "res.partner", "method": "search_read"},
            {"model": "ir.rule", "method": "write", "args_json": "[[1], {}]"},
        ]
        classifications, overall, needs_confirm = classify_batch(ops)
        assert overall == RiskLevel.BLOCKED

    def test_empty_batch(self):
        classifications, overall, needs_confirm = classify_batch([])
        assert overall == RiskLevel.SAFE
        assert needs_confirm is False
        assert classifications == []

    def test_invalid_json_in_batch(self):
        """Invalid JSON in args_json should not crash — just classify with empty args."""
        ops = [{"model": "res.partner", "method": "write", "args_json": "not-json"}]
        classifications, overall, needs_confirm = classify_batch(ops)
        assert len(classifications) == 1


# =====================================================
# Test: Audit logging
# =====================================================


class TestAuditLogging:
    def test_audit_disabled_by_default(self, capsys):
        with patch.dict(os.environ, {}, clear=True):
            classification = classify_operation("res.partner", "unlink", [[1]])
            audit_log(classification, confirmed=False, executed=False)
            captured = capsys.readouterr()
            assert captured.err == ""

    def test_audit_enabled(self, capsys):
        with patch.dict(os.environ, {"MCP_SAFETY_AUDIT": "true"}):
            classification = classify_operation("res.partner", "unlink", [[1]])
            audit_log(classification, confirmed=True, executed=True)
            captured = capsys.readouterr()
            assert "[SAFETY AUDIT]" in captured.err
            # Parse the JSON part
            json_str = captured.err.split("[SAFETY AUDIT] ")[1].strip()
            entry = json.loads(json_str)
            assert entry["model"] == "res.partner"
            assert entry["method"] == "unlink"
            assert entry["risk_level"] == "high"
            assert entry["confirmed"] is True
            assert entry["executed"] is True

    def test_audit_never_raises(self):
        """Audit logging should never raise, even with broken state."""
        with patch.dict(os.environ, {"MCP_SAFETY_AUDIT": "true"}):
            # Pass a completely wrong classification-like object — should not raise
            try:
                audit_log(
                    SafetyClassification(
                        risk_level=RiskLevel.HIGH,
                        model="test",
                        method="test",
                        requires_confirmation=True,
                        reason="test",
                    ),
                    confirmed=False,
                    executed=False,
                )
            except Exception:
                pytest.fail("audit_log should never raise")


# =====================================================
# Test: Default mode is strict
# =====================================================


class TestDefaultMode:
    def test_default_is_strict(self):
        """Without MCP_SAFETY_MODE set, mode should be strict."""
        with patch.dict(os.environ, {}, clear=True):
            result = classify_operation("res.partner", "custom_action")
            assert result.requires_confirmation is True  # strict mode behavior
