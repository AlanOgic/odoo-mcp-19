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

    @pytest.mark.parametrize("method", ["generate", "revoke"])
    def test_api_key_management_is_blocked(self, method):
        """Odoo 19.1+ programmatic API-key methods on res.users.apikeys must be
        BLOCKED — minting a key is a persistent privilege escalation."""
        result = classify_operation(
            "res.users.apikeys",
            method,
            [{"scope": None, "name": "x", "expiration_date": "2027-01-01"}],
        )
        assert result.risk_level == RiskLevel.BLOCKED
        assert result.blocked_reason is not None


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
    """MEDIUM_METHODS in strict mode: every side-effect call is gated, whatever the record count."""

    @pytest.fixture(autouse=True)
    def set_strict_mode(self):
        with patch.dict(os.environ, {"MCP_SAFETY_MODE": "strict"}):
            yield

    @pytest.mark.parametrize("model", ["res.partner", "crm.lead", "sale.order", "product.product"])
    def test_single_record_write_requires_confirm(self, model):
        """A write on one real quotation / opportunity / customer must never execute from a single call."""
        result = classify_operation(model, "write", [[1], {"name": "x"}])
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.requires_confirmation is True
        assert result.record_count == 1

    def test_single_create_requires_confirm(self):
        result = classify_operation("res.partner", "create", [{"name": "x"}])
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.requires_confirmation is True
        assert result.record_count == 1

    @pytest.mark.parametrize("method", ["copy", "name_create", "load"])
    def test_other_medium_methods_require_confirm(self, method):
        assert classify_operation("res.partner", method, [[1]]).requires_confirmation is True

    def test_batch_of_single_record_writes_is_gated(self):
        """batch_execute gates on the same classification — two ungated ops would slip a batch through."""
        ops = [
            {"model": "res.partner", "method": "write", "args_json": '[[1], {"name": "a"}]'},
            {"model": "crm.lead", "method": "write", "args_json": '[[7], {"probability": 0}]'},
        ]
        _, overall, any_needs_confirmation = classify_batch(ops)
        assert overall == RiskLevel.MEDIUM
        assert any_needs_confirmation is True

    def test_multi_record_write_requires_confirm(self):
        """Multi-record write in strict DOES require confirmation."""
        result = classify_operation("res.partner", "write", [[1, 2, 3], {"name": "x"}])
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.requires_confirmation is True
        assert result.record_count == 3

    def test_batch_create_requires_confirm(self):
        result = classify_operation("res.partner", "create", [[{"name": "a"}, {"name": "b"}]])
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
        result = classify_operation("res.partner", "create", [[{"name": "a"}, {"name": "b"}]])
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
            "res.partner",
            "create",
            [[{"name": "a"}, {"name": "b"}, {"name": "c"}]],
        )
        assert result.record_count == 3

    def test_action_with_ids(self):
        result = classify_operation("sale.order", "action_confirm", [[1, 2]])
        assert result.record_count == 2

    def test_action_with_single_id(self):
        result = classify_operation("sale.order", "action_confirm", [1])
        assert result.record_count == 1

    def test_copy_single_id(self):
        result = classify_operation("res.partner", "copy", [1])
        assert result.record_count == 1

    def test_no_args(self):
        result = classify_operation("res.partner", "write")
        assert result.record_count is None


# =====================================================
# Test: Record counting via named (kwargs) arguments
# =====================================================


class TestNamedArgRecordCounting:
    """The recordset may arrive in args_json OR kwargs_json.

    JSON-2 is named-args-only, so `write(ids=[1,2,3], vals={...})` and
    `write([1,2,3], {...})` are the same call. Counting only the positional
    form let a bulk operation slip past the strict-mode confirmation gate by
    simply moving the ids into kwargs_json.
    """

    def test_write_ids_in_kwargs_counts(self):
        result = classify_operation("res.partner", "write", [], {"ids": [1, 2, 3], "vals": {"name": "x"}})
        assert result.record_count == 3

    def test_write_batch_in_kwargs_requires_confirmation(self):
        with patch.dict(os.environ, {"MCP_SAFETY_MODE": "strict"}):
            result = classify_operation("res.partner", "write", [], {"ids": [1, 2, 3], "vals": {"name": "x"}})
        assert result.requires_confirmation is True

    def test_positional_and_named_forms_agree(self):
        """Both spellings of the same call must classify identically."""
        with patch.dict(os.environ, {"MCP_SAFETY_MODE": "strict"}):
            positional = classify_operation("res.partner", "write", [[1, 2, 3], {"name": "x"}])
            named = classify_operation("res.partner", "write", [], {"ids": [1, 2, 3], "vals": {"name": "x"}})
        assert positional.record_count == named.record_count
        assert positional.requires_confirmation == named.requires_confirmation

    def test_create_vals_list_in_kwargs_counts(self):
        result = classify_operation("res.partner", "create", [], {"vals_list": [{"name": "a"}, {"name": "b"}]})
        assert result.record_count == 2

    def test_action_ids_in_kwargs_counts(self):
        result = classify_operation("sale.order", "action_confirm", [], {"ids": [1, 2, 3, 4]})
        assert result.record_count == 4

    def test_load_data_rows_count(self):
        """`load` bulk-imports rows; its payload is the `data` argument."""
        rows = [["1", f"name{i}"] for i in range(50)]
        with patch.dict(os.environ, {"MCP_SAFETY_MODE": "strict"}):
            result = classify_operation("res.partner", "load", [], {"fields": ["id", "name"], "data": rows})
        assert result.record_count == 50
        assert result.requires_confirmation is True

    def test_copy_batch_in_kwargs_counts(self):
        result = classify_operation("res.partner", "copy", [], {"ids": [1, 2, 3]})
        assert result.record_count == 3

    def test_positional_wins_over_named(self):
        """A positional recordset is what arg_mapping forwards; prefer it."""
        result = classify_operation("res.partner", "write", [[1, 2], {"name": "x"}], {"ids": [1, 2, 3, 4, 5]})
        assert result.record_count == 2

    def test_unrelated_kwargs_do_not_count(self):
        result = classify_operation("res.partner", "search_read", [], {"domain": [], "fields": ["a", "b", "c"]})
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
    def test_removed_quote_to_cash_returns_none(self):
        # quote_to_cash and its aliases were removed post-1.15.0
        for name in ["quote_to_cash", "quotation_to_invoice", "sales_workflow"]:
            assert classify_workflow(name) is None

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
        preview = classify_workflow("Lead_To_Won")
        assert preview is not None

    def test_cascade_warnings_in_workflow(self):
        preview = classify_workflow("create_and_post_invoice")
        assert preview is not None
        warnings = [s.cascade_warning for s in preview.steps if s.cascade_warning]
        assert len(warnings) > 0  # account.move action_post is irreversible


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
    def test_audit_disabled_by_default(self, caplog):
        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level("INFO", logger="odoo_mcp.safety"):
                classification = classify_operation("res.partner", "unlink", [[1]])
                audit_log(classification, confirmed=False, executed=False)
            assert not any("[SAFETY AUDIT]" in r.message for r in caplog.records)

    def test_audit_enabled(self, caplog):
        with patch.dict(os.environ, {"MCP_SAFETY_AUDIT": "true"}):
            with caplog.at_level("INFO", logger="odoo_mcp.safety"):
                classification = classify_operation("res.partner", "unlink", [[1]])
                audit_log(classification, confirmed=True, executed=True)
            audit_records = [r for r in caplog.records if "[SAFETY AUDIT]" in r.message]
            assert len(audit_records) == 1
            # The message format is "[SAFETY AUDIT] {json}" — parse the JSON
            json_str = audit_records[0].message.split("[SAFETY AUDIT] ", 1)[1]
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
