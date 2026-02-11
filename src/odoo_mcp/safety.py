"""
Safety classification layer for Odoo MCP Server.

Pre-execution safety checks that classify operations by risk level
and gate dangerous operations behind confirmation. Zero FastMCP dependency.

Environment variables:
    MCP_SAFETY_MODE: 'strict' (default) or 'permissive'
    MCP_SAFETY_AUDIT: 'true' to enable audit logging to stderr
"""

import json
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ----- Risk Levels -----

class RiskLevel(str, Enum):
    """Risk classification for Odoo operations."""
    SAFE = "safe"           # Execute immediately, no confirmation
    MEDIUM = "medium"       # Gate based on mode/volume
    HIGH = "high"           # Always require confirmation
    BLOCKED = "blocked"     # Always refuse


# ----- Method Classification Sets -----

SAFE_METHODS = frozenset({
    "search_read", "read", "search", "search_count",
    "fields_get", "name_get", "name_search", "default_get",
    "read_group", "formatted_read_group",
    "has_access", "check_access_rights", "export_data",
})

MEDIUM_METHODS = frozenset({
    "create", "write", "copy", "name_create", "load",
})

HIGH_METHODS = frozenset({
    "unlink",
    "action_confirm", "action_cancel", "action_done",
    "action_draft", "action_validate", "action_post",
    "action_assign", "action_set_won", "action_set_lost",
    "button_confirm", "button_cancel", "button_draft",
    "button_validate",
})


# ----- Model Classifications -----

BLOCKED_MODELS = frozenset({
    "ir.rule",
    "ir.model.access",
    "ir.module.module",
    "ir.config_parameter",
    "res.users",
})

SENSITIVE_MODELS = frozenset({
    "account.move",
    "account.payment",
    "account.bank.statement",
    "hr.payslip",
    "ir.cron",
})


# ----- Cascade Warnings -----

CASCADE_WARNINGS: Dict[Tuple[str, str], str] = {
    ("sale.order", "action_confirm"): (
        "Confirming a sales order creates delivery orders and "
        "may trigger procurement rules."
    ),
    ("account.move", "action_post"): (
        "Posting a journal entry creates accounting entries. "
        "This is generally irreversible without a reversal entry."
    ),
    ("stock.picking", "button_validate"): (
        "Validating a transfer updates stock levels and creates "
        "stock valuation entries."
    ),
    ("purchase.order", "button_confirm"): (
        "Confirming a purchase order creates incoming receipts "
        "and may trigger supplier notifications."
    ),
    ("account.payment", "action_post"): (
        "Posting a payment creates journal entries and triggers "
        "automatic reconciliation."
    ),
}


# ----- Pydantic Models -----

class SafetyClassification(BaseModel):
    """Result of classifying an operation's risk level."""
    risk_level: RiskLevel = Field(description="Classified risk level")
    model: str = Field(description="Odoo model name")
    method: str = Field(description="Method name")
    record_count: Optional[int] = Field(
        default=None, description="Estimated number of records affected"
    )
    requires_confirmation: bool = Field(
        description="Whether the caller must re-call with confirmed=true"
    )
    reason: str = Field(description="Human-readable reason for the classification")
    cascade_warning: Optional[str] = Field(
        default=None, description="Warning about side effects"
    )
    blocked_reason: Optional[str] = Field(
        default=None, description="Reason when operation is blocked"
    )


class PendingConfirmation(BaseModel):
    """Returned when an operation needs confirmation before execution."""
    pending_confirmation: bool = Field(default=True)
    classification: SafetyClassification
    message: str = Field(description="User-facing confirmation message")
    confirm_hint: str = Field(
        default="Re-call with confirmed=true to proceed.",
        description="Hint for the caller",
    )


class WorkflowStepClassification(BaseModel):
    """Classification for a single workflow step."""
    step: str = Field(description="Step name")
    model: str = Field(description="Model involved")
    method: str = Field(description="Method called")
    risk_level: RiskLevel = Field(description="Risk level for this step")
    cascade_warning: Optional[str] = Field(default=None)


class WorkflowSafetyPreview(BaseModel):
    """Safety preview for a complete workflow."""
    pending_confirmation: bool = Field(default=True)
    workflow: str = Field(description="Workflow name")
    steps: List[WorkflowStepClassification] = Field(
        description="Classification for each step"
    )
    overall_risk: RiskLevel = Field(description="Highest risk across all steps")
    message: str = Field(description="User-facing summary")


# ----- Helpers -----

def _get_safety_mode() -> str:
    """Get the configured safety mode."""
    return os.environ.get("MCP_SAFETY_MODE", "strict").lower()


def _estimate_record_count(method: str, args: list, kwargs: dict) -> Optional[int]:
    """Estimate the number of records affected by an operation."""
    try:
        if method in ("unlink", "write"):
            # First arg is list of IDs
            if args and isinstance(args[0], list):
                return len(args[0])
        elif method == "create":
            # First arg is vals dict or list of vals dicts
            if args:
                val = args[0]
                if isinstance(val, list):
                    return len(val)
                return 1
        elif method.startswith("action_") or method.startswith("button_"):
            # First arg is list of IDs
            if args and isinstance(args[0], list):
                return len(args[0])
            elif args and isinstance(args[0], int):
                return 1
        elif method == "copy":
            return 1
    except (IndexError, TypeError):
        pass
    return None


# ----- Core Classification -----

def classify_operation(
    model: str,
    method: str,
    args: Optional[list] = None,
    kwargs: Optional[dict] = None,
) -> SafetyClassification:
    """
    Classify an Odoo operation by risk level.

    Classification logic:
    1. SAFE_METHODS → SAFE (even on blocked/sensitive models)
    2. BLOCKED_MODELS + non-safe method → BLOCKED
    3. HIGH_METHODS → HIGH (always confirm)
    4. MEDIUM_METHODS → depends on mode/model/volume
    5. Unknown methods → MEDIUM
    """
    args = args or []
    kwargs = kwargs or {}
    mode = _get_safety_mode()
    record_count = _estimate_record_count(method, args, kwargs)
    cascade_warning = CASCADE_WARNINGS.get((model, method))

    # 1. Safe methods are always safe, regardless of model
    if method in SAFE_METHODS:
        return SafetyClassification(
            risk_level=RiskLevel.SAFE,
            model=model,
            method=method,
            record_count=record_count,
            requires_confirmation=False,
            reason="Read-only or safe method.",
        )

    # 2. Blocked models refuse all non-safe methods
    if model in BLOCKED_MODELS:
        return SafetyClassification(
            risk_level=RiskLevel.BLOCKED,
            model=model,
            method=method,
            record_count=record_count,
            requires_confirmation=False,
            reason=f"Model '{model}' is a security-critical model.",
            blocked_reason=(
                f"Write operations on '{model}' are blocked for safety. "
                f"Use the Odoo web interface to modify security settings."
            ),
        )

    # 3. High-risk methods always require confirmation
    if method in HIGH_METHODS:
        reason = f"'{method}' is a high-risk operation"
        if record_count and record_count > 1:
            reason += f" affecting {record_count} records"
        reason += "."
        return SafetyClassification(
            risk_level=RiskLevel.HIGH,
            model=model,
            method=method,
            record_count=record_count,
            requires_confirmation=True,
            reason=reason,
            cascade_warning=cascade_warning,
        )

    # 4. Medium-risk methods: depends on mode, model, volume
    if method in MEDIUM_METHODS:
        # Sensitive models always need confirmation for writes
        if model in SENSITIVE_MODELS:
            return SafetyClassification(
                risk_level=RiskLevel.MEDIUM,
                model=model,
                method=method,
                record_count=record_count,
                requires_confirmation=True,
                reason=(
                    f"'{method}' on sensitive model '{model}' "
                    f"requires confirmation."
                ),
                cascade_warning=cascade_warning,
            )

        # Strict mode: confirm if batch (record_count > 1)
        if mode == "strict" and record_count is not None and record_count > 1:
            return SafetyClassification(
                risk_level=RiskLevel.MEDIUM,
                model=model,
                method=method,
                record_count=record_count,
                requires_confirmation=True,
                reason=(
                    f"'{method}' affects {record_count} records "
                    f"(strict mode requires confirmation for batch operations)."
                ),
                cascade_warning=cascade_warning,
            )

        # Otherwise: safe to proceed
        return SafetyClassification(
            risk_level=RiskLevel.MEDIUM,
            model=model,
            method=method,
            record_count=record_count,
            requires_confirmation=False,
            reason=f"'{method}' classified as medium risk, no confirmation needed.",
            cascade_warning=cascade_warning,
        )

    # 5. Unknown methods → MEDIUM, confirmation depends on mode
    requires_confirm = mode == "strict"
    return SafetyClassification(
        risk_level=RiskLevel.MEDIUM,
        model=model,
        method=method,
        record_count=record_count,
        requires_confirmation=requires_confirm,
        reason=(
            f"Unknown method '{method}' — "
            f"{'confirmation required in strict mode' if requires_confirm else 'allowed in permissive mode'}."
        ),
        cascade_warning=cascade_warning,
    )


# ----- Batch Classification -----

def classify_batch(
    operations: List[Dict[str, Any]],
) -> Tuple[List[SafetyClassification], RiskLevel, bool]:
    """
    Classify all operations in a batch.

    Returns:
        Tuple of (classifications, overall_risk_level, any_needs_confirmation)
    """
    classifications = []
    overall_risk = RiskLevel.SAFE
    any_needs_confirmation = False

    risk_order = {
        RiskLevel.SAFE: 0,
        RiskLevel.MEDIUM: 1,
        RiskLevel.HIGH: 2,
        RiskLevel.BLOCKED: 3,
    }

    for op in operations:
        model = op.get("model", "unknown")
        method = op.get("method", "unknown")

        args = []
        kwargs = {}
        try:
            if op.get("args_json"):
                args = json.loads(op["args_json"])
            if op.get("kwargs_json"):
                kwargs = json.loads(op["kwargs_json"])
        except (json.JSONDecodeError, TypeError):
            pass

        classification = classify_operation(model, method, args, kwargs)
        classifications.append(classification)

        if risk_order[classification.risk_level] > risk_order[overall_risk]:
            overall_risk = classification.risk_level

        if classification.requires_confirmation:
            any_needs_confirmation = True

    return classifications, overall_risk, any_needs_confirmation


# ----- Workflow Classification -----

# Maps workflow name → list of (step_name, model, method)
_WORKFLOW_STEPS: Dict[str, List[Tuple[str, str, str]]] = {
    "quote_to_cash": [
        ("confirm_order", "sale.order", "action_confirm"),
        ("create_invoice", "sale.order", "_create_invoices"),
        ("post_invoice", "account.move", "action_post"),
    ],
    "quotation_to_invoice": [
        ("confirm_order", "sale.order", "action_confirm"),
        ("create_invoice", "sale.order", "_create_invoices"),
        ("post_invoice", "account.move", "action_post"),
    ],
    "sales_workflow": [
        ("confirm_order", "sale.order", "action_confirm"),
        ("create_invoice", "sale.order", "_create_invoices"),
        ("post_invoice", "account.move", "action_post"),
    ],
    "lead_to_won": [
        ("convert_to_opportunity", "crm.lead", "convert_opportunity"),
        ("mark_won", "crm.lead", "action_set_won"),
    ],
    "crm_workflow": [
        ("convert_to_opportunity", "crm.lead", "convert_opportunity"),
        ("mark_won", "crm.lead", "action_set_won"),
    ],
    "opportunity_won": [
        ("convert_to_opportunity", "crm.lead", "convert_opportunity"),
        ("mark_won", "crm.lead", "action_set_won"),
    ],
    "create_and_post_invoice": [
        ("create_invoice", "account.move", "create"),
        ("post_invoice", "account.move", "action_post"),
    ],
    "quick_invoice": [
        ("create_invoice", "account.move", "create"),
        ("post_invoice", "account.move", "action_post"),
    ],
    "stock_transfer": [
        ("confirm_transfer", "stock.picking", "action_confirm"),
        ("validate_transfer", "stock.picking", "button_validate"),
    ],
}


def classify_workflow(
    workflow: str,
    params: Optional[dict] = None,
) -> Optional[WorkflowSafetyPreview]:
    """
    Classify a workflow by its name.

    Returns None for unknown workflows (let the caller handle them).
    """
    workflow_lower = workflow.lower().strip()
    steps_def = _WORKFLOW_STEPS.get(workflow_lower)

    if steps_def is None:
        return None

    risk_order = {
        RiskLevel.SAFE: 0,
        RiskLevel.MEDIUM: 1,
        RiskLevel.HIGH: 2,
        RiskLevel.BLOCKED: 3,
    }

    step_classifications = []
    overall_risk = RiskLevel.SAFE

    for step_name, model, method in steps_def:
        classification = classify_operation(model, method)
        cascade_warning = CASCADE_WARNINGS.get((model, method))

        step_cls = WorkflowStepClassification(
            step=step_name,
            model=model,
            method=method,
            risk_level=classification.risk_level,
            cascade_warning=cascade_warning,
        )
        step_classifications.append(step_cls)

        if risk_order[classification.risk_level] > risk_order[overall_risk]:
            overall_risk = classification.risk_level

    # Build human-readable message
    high_steps = [
        s for s in step_classifications
        if s.risk_level in (RiskLevel.HIGH, RiskLevel.BLOCKED)
    ]
    warnings = [s.cascade_warning for s in step_classifications if s.cascade_warning]

    message_parts = [
        f"Workflow '{workflow}' contains {len(step_classifications)} steps "
        f"with overall risk level: {overall_risk.value}."
    ]
    if high_steps:
        message_parts.append(
            f"High-risk steps: {', '.join(s.step for s in high_steps)}."
        )
    if warnings:
        message_parts.append("Side effects: " + " | ".join(warnings))

    return WorkflowSafetyPreview(
        workflow=workflow,
        steps=step_classifications,
        overall_risk=overall_risk,
        message=" ".join(message_parts),
    )


# ----- Audit Logger -----

def _is_audit_enabled() -> bool:
    """Check if audit logging is enabled."""
    return os.environ.get("MCP_SAFETY_AUDIT", "").lower() == "true"


def audit_log(
    classification: SafetyClassification,
    confirmed: bool,
    executed: bool,
) -> None:
    """
    Write an audit log entry to stderr. Silent on failure.

    Only active when MCP_SAFETY_AUDIT=true.
    """
    if not _is_audit_enabled():
        return

    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "safety_audit",
            "model": classification.model,
            "method": classification.method,
            "risk_level": classification.risk_level.value,
            "record_count": classification.record_count,
            "requires_confirmation": classification.requires_confirmation,
            "confirmed": confirmed,
            "executed": executed,
        }
        if classification.cascade_warning:
            entry["cascade_warning"] = classification.cascade_warning
        if classification.blocked_reason:
            entry["blocked_reason"] = classification.blocked_reason

        print(
            f"[SAFETY AUDIT] {json.dumps(entry)}",
            file=sys.stderr,
        )
    except Exception:
        pass  # Audit logging never raises
