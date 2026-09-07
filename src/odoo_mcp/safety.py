"""
Safety classification layer for Odoo MCP Server.

Pre-execution safety checks that classify operations by risk level
and gate dangerous operations behind confirmation. Zero FastMCP dependency.

Environment variables:
    MCP_SAFETY_MODE: 'strict' (default) or 'permissive'
    MCP_SAFETY_AUDIT: 'true' to enable audit logging to stderr
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ----- Risk Levels -----


class RiskLevel(str, Enum):
    """Risk classification for Odoo operations."""

    SAFE = "safe"  # Execute immediately, no confirmation
    MEDIUM = "medium"  # Gate based on mode/volume
    HIGH = "high"  # Always require confirmation
    BLOCKED = "blocked"  # Always refuse


# ----- Method Classification Sets -----

SAFE_METHODS = frozenset(
    {
        "search_read",
        "read",
        "search",
        "search_count",
        "fields_get",
        "name_get",
        "name_search",
        "default_get",
        "read_group",
        "formatted_read_group",
        "has_access",
        "check_access_rights",
        "export_data",
    }
)

MEDIUM_METHODS = frozenset(
    {
        "create",
        "write",
        "copy",
        "name_create",
        "load",
    }
)

HIGH_METHODS = frozenset(
    {
        "unlink",
        "action_confirm",
        "action_cancel",
        "action_done",
        "action_draft",
        "action_validate",
        "action_post",
        "action_assign",
        "action_set_won",
        "action_set_lost",
        "button_confirm",
        "button_cancel",
        "button_draft",
        "button_validate",
    }
)


# ----- Model Classifications -----

BLOCKED_MODELS = frozenset(
    {
        "ir.rule",
        "ir.model.access",
        "ir.module.module",
        "ir.config_parameter",
        "ir.model",
        "res.users",
        "res.groups",
        # Odoo 19.1+ exposes programmatic API-key management via JSON-2
        # (res.users.apikeys.generate / .revoke). A distinct model name from
        # res.users, so it must be listed explicitly — otherwise an agent could
        # mint a persistent, unscoped API key that outlives the MCP session
        # (privilege escalation / backdoor). No legitimate agent flow needs it.
        "res.users.apikeys",
    }
)

SENSITIVE_MODELS = frozenset(
    {
        "account.move",
        "account.payment",
        "account.bank.statement",
        "hr.payslip",
        "ir.cron",
        # Studio-style schema customization: creating/editing custom fields is
        # allowed but always requires explicit confirmation (token gate), in both
        # strict and permissive modes. Whole-model changes (ir.model) stay BLOCKED.
        "ir.model.fields",
    }
)


# ----- Cascade Warnings -----

CASCADE_WARNINGS: dict[tuple[str, str], str] = {
    ("sale.order", "action_confirm"): (
        "Confirming a sales order creates delivery orders and " "may trigger procurement rules."
    ),
    ("account.move", "action_post"): (
        "Posting a journal entry creates accounting entries. "
        "This is generally irreversible without a reversal entry."
    ),
    ("stock.picking", "button_validate"): (
        "Validating a transfer updates stock levels and creates " "stock valuation entries."
    ),
    ("purchase.order", "button_confirm"): (
        "Confirming a purchase order creates incoming receipts " "and may trigger supplier notifications."
    ),
    ("account.payment", "action_post"): (
        "Posting a payment creates journal entries and triggers " "automatic reconciliation."
    ),
}


# ----- Side-Effect Method Predicate -----

# Modes whose classifier behaviour requires confirmation for unknown methods
# and for batch (record_count > 1) MEDIUM operations. "locked" inherits the
# "strict" classifier semantics in addition to its own profile-layer gates
# (read_only, write_allowlist, validate_payloads).
_STRICT_EQUIV: frozenset[str] = frozenset({"strict", "locked"})


def is_side_effect_method(method: str) -> bool:
    """Return True if calling this method must be treated as a side effect.

    Single source of truth for the read-only guard, the write allowlist, and
    the payload pre-flight. Cheap set lookup — does NOT call the classifier.

    **Fail-closed: anything outside SAFE_METHODS counts as a side effect.**
    Recognising writes by name shape (a literal CRUD set plus action_* /
    button_* prefixes) is not sufficient for a kill-switch, because Odoo is
    full of write methods that match neither. `module_knowledge.json` alone
    documents `add_members`, `article_create`, `article_duplicate`,
    `channel_create`, `convert_opportunity`, `create_from_urls`,
    `create_from_attachments`, `create_from_binary_files`, `document_create`,
    `get_direct_response` and `open_agent_chat`; the standard ORM adds
    ubiquitous ones like `message_post` and `toggle_active`. Under
    MCP_READ_ONLY none of those may reach Odoo, so the only defensible
    default is to gate everything we cannot prove is a read.

    SAFE_METHODS (search_read, read, fields_get, ...) return False.
    """
    return method not in SAFE_METHODS


def _allowlist_blocks(model: str, method: str, profile) -> bool:
    """Return True if the resolved profile's allowlist is enforced AND the
    given (model, method) is not permitted.

    Does NOT short-circuit BLOCKED_MODELS or SAFE methods — callers must
    check those first.
    """
    if not profile.write_allowlist_enforced:
        return False
    if not is_side_effect_method(method):
        return False
    full_key = f"{model}.{method}"
    wildcard_key = f"{model}.*"
    return full_key not in profile.write_allowlist and wildcard_key not in profile.write_allowlist


# ----- Pydantic Models -----


class SafetyClassification(BaseModel):
    """Result of classifying an operation's risk level."""

    risk_level: RiskLevel = Field(description="Classified risk level")
    model: str = Field(description="Odoo model name")
    method: str = Field(description="Method name")
    record_count: int | None = Field(default=None, description="Estimated number of records affected")
    requires_confirmation: bool = Field(description="Whether the caller must re-call with confirmed=true")
    reason: str = Field(description="Human-readable reason for the classification")
    cascade_warning: str | None = Field(default=None, description="Warning about side effects")
    blocked_reason: str | None = Field(default=None, description="Reason when operation is blocked")


class WorkflowStepClassification(BaseModel):
    """Classification for a single workflow step."""

    step: str = Field(description="Step name")
    model: str = Field(description="Model involved")
    method: str = Field(description="Method called")
    risk_level: RiskLevel = Field(description="Risk level for this step")
    cascade_warning: str | None = Field(default=None)


class WorkflowSafetyPreview(BaseModel):
    """Safety preview for a complete workflow."""

    pending_confirmation: bool = Field(default=True)
    workflow: str = Field(description="Workflow name")
    steps: list[WorkflowStepClassification] = Field(description="Classification for each step")
    overall_risk: RiskLevel = Field(description="Highest risk across all steps")
    message: str = Field(description="User-facing summary")


_RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.SAFE: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.BLOCKED: 3,
}


# ----- Helpers -----


def _get_safety_mode() -> str:
    """Get the configured safety mode (read from env on each call)."""
    return os.environ.get("MCP_SAFETY_MODE", "strict").lower()


# Which argument carries the payload whose size determines the record count,
# given as (positional index, JSON-2 named form). Both spellings must be
# consulted: the v2 API is named-args-only (see arg_mapping), so an agent can
# express the same call either positionally in args_json or by name in
# kwargs_json. Counting only the positional form would let a bulk operation
# slip past the strict-mode confirmation gate by moving ids into kwargs_json.
_COUNTED_ARG: dict[str, tuple[int, str]] = {
    "write": (0, "ids"),
    "unlink": (0, "ids"),
    "copy": (0, "ids"),
    "create": (0, "vals_list"),
    "load": (1, "data"),
}

# action_* / button_* run on a recordset passed the same way (arg_mapping
# routes position 0 to "ids" for them, including the generic fallback).
_RECORD_BOUND_ARG: tuple[int, str] = (0, "ids")


def _counted_argument(method: str, args: list, kwargs: dict) -> Any:
    """Return the argument whose size determines the operation's record count."""
    spec = _COUNTED_ARG.get(method)
    if spec is None and method.startswith(("action_", "button_")):
        spec = _RECORD_BOUND_ARG
    if spec is None:
        return None
    position, name = spec
    if position < len(args):
        return args[position]
    return kwargs.get(name)


def _estimate_record_count(method: str, args: list, kwargs: dict) -> int | None:
    """Estimate the number of records affected by an operation.

    Reads the recordset from args_json or kwargs_json — both are valid ways to
    express the same JSON-2 call, so counting only one of them would leave the
    strict-mode batch gate bypassable by choosing the other form.
    """
    payload = _counted_argument(method, args, kwargs)
    if isinstance(payload, list):
        return len(payload)
    # A bare id or a single vals dict is one record. bool is an int subclass,
    # so exclude it rather than counting True as a record.
    if isinstance(payload, (dict, int)) and not isinstance(payload, bool):
        return 1
    return None


# ----- Core Classification -----


def classify_operation(
    model: str,
    method: str,
    args: list | None = None,
    kwargs: dict | None = None,
    role: str | None = None,
) -> SafetyClassification:
    """
    Classify an Odoo operation by risk level.

    Classification logic:
    1. SAFE_METHODS → SAFE (even on blocked/sensitive models)
    2. BLOCKED_MODELS + non-safe method → BLOCKED
    3. Allowlist enforcement — side-effect calls blocked unless explicitly permitted
    4. HIGH_METHODS → HIGH (always confirm)
    5. MEDIUM_METHODS → depends on mode/model/volume
    6. Unknown methods → MEDIUM
    """
    args = args or []
    kwargs = kwargs or {}
    from .safety_profile import get_profile

    profile = get_profile()
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

    # 1b. Read-only profiles: anything beyond safe methods is blocked.
    if role == "readonly":
        return SafetyClassification(
            risk_level=RiskLevel.BLOCKED,
            model=model,
            method=method,
            record_count=record_count,
            requires_confirmation=False,
            reason="Read-only profile: write operations are blocked.",
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

    # 3. Allowlist enforcement — explicit permits required for side-effect calls.
    if _allowlist_blocks(model, method, profile):
        return SafetyClassification(
            risk_level=RiskLevel.BLOCKED,
            model=model,
            method=method,
            record_count=record_count,
            requires_confirmation=False,
            reason=f"'{model}.{method}' is not in MCP_WRITE_ALLOWLIST.",
            blocked_reason=(
                f"Side-effect call '{model}.{method}' rejected: not present in "
                f"MCP_WRITE_ALLOWLIST. Add the entry to allow it, or use a "
                f"safe read method instead."
            ),
        )

    # 4. High-risk methods always require confirmation
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

    # 5. Medium-risk methods: depends on mode, model, volume
    if method in MEDIUM_METHODS:
        # Sensitive models always need confirmation for writes
        if model in SENSITIVE_MODELS:
            return SafetyClassification(
                risk_level=RiskLevel.MEDIUM,
                model=model,
                method=method,
                record_count=record_count,
                requires_confirmation=True,
                reason=(f"'{method}' on sensitive model '{model}' " f"requires confirmation."),
                cascade_warning=cascade_warning,
            )

        # Modes treated as confirmation-required ("strict" semantics).
        # Note: "locked" inherits "strict" classifier behaviour (in addition to
        # its own read_only / allowlist gates resolved at the profile layer).
        # Every side-effect call is gated, whatever the record count: before
        # v1.18.1 a single-record write/create on a non-sensitive model ran
        # unconfirmed from one tool call, and batch_execute inherited that.
        if mode in _STRICT_EQUIV:
            scope = f"{record_count} record(s)" if record_count is not None else "an unknown number of records"
            return SafetyClassification(
                risk_level=RiskLevel.MEDIUM,
                model=model,
                method=method,
                record_count=record_count,
                requires_confirmation=True,
                reason=f"'{method}' on '{model}' affects {scope} (strict mode confirms every write).",
                cascade_warning=cascade_warning,
            )

        # Permissive mode: medium-risk writes proceed without a gate.
        return SafetyClassification(
            risk_level=RiskLevel.MEDIUM,
            model=model,
            method=method,
            record_count=record_count,
            requires_confirmation=False,
            reason=f"'{method}' classified as medium risk, no confirmation needed in permissive mode.",
            cascade_warning=cascade_warning,
        )

    # 6. Unknown methods → MEDIUM, confirmation depends on mode
    requires_confirm = mode in _STRICT_EQUIV
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
    operations: list[dict[str, Any]],
    role: str | None = None,
) -> tuple[list[SafetyClassification], RiskLevel, bool]:
    """
    Classify all operations in a batch.

    Returns:
        Tuple of (classifications, overall_risk_level, any_needs_confirmation)
    """
    classifications = []
    overall_risk = RiskLevel.SAFE
    any_needs_confirmation = False

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
        except (json.JSONDecodeError, TypeError) as exc:
            # Classification proceeds on empty args (the record-count gate then
            # cannot under-count); execution will reject the same payload later.
            logger.warning(
                "Batch op %s.%s has malformed args_json/kwargs_json, classifying without payload: %s",
                model,
                method,
                exc,
            )

        classification = classify_operation(model, method, args, kwargs, role=role)
        classifications.append(classification)

        if _RISK_ORDER[classification.risk_level] > _RISK_ORDER[overall_risk]:
            overall_risk = classification.risk_level

        if classification.requires_confirmation:
            any_needs_confirmation = True

    return classifications, overall_risk, any_needs_confirmation


# ----- Workflow Classification -----

# Canonical step lists (defined once, aliased below)
_LEAD_TO_WON_STEPS: list[tuple[str, str, str]] = [
    ("convert_to_opportunity", "crm.lead", "convert_opportunity"),
    ("mark_won", "crm.lead", "action_set_won"),
]

_CREATE_AND_POST_INVOICE_STEPS: list[tuple[str, str, str]] = [
    ("create_invoice", "account.move", "create"),
    ("post_invoice", "account.move", "action_post"),
]

_STOCK_TRANSFER_STEPS: list[tuple[str, str, str]] = [
    ("confirm_transfer", "stock.picking", "action_confirm"),
    ("validate_transfer", "stock.picking", "button_validate"),
]

# Maps workflow name → list of (step_name, model, method)
_WORKFLOW_STEPS: dict[str, list[tuple[str, str, str]]] = {
    "lead_to_won": _LEAD_TO_WON_STEPS,
    "crm_workflow": _LEAD_TO_WON_STEPS,
    "opportunity_won": _LEAD_TO_WON_STEPS,
    "create_and_post_invoice": _CREATE_AND_POST_INVOICE_STEPS,
    "quick_invoice": _CREATE_AND_POST_INVOICE_STEPS,
    "stock_transfer": _STOCK_TRANSFER_STEPS,
}


def classify_workflow(
    workflow: str,
    params: dict | None = None,
    role: str | None = None,
) -> WorkflowSafetyPreview | None:
    """
    Classify a workflow by its name.

    Returns None for unknown workflows (let the caller handle them).
    """
    workflow_lower = workflow.lower().strip()
    steps_def = _WORKFLOW_STEPS.get(workflow_lower)

    if steps_def is None:
        return None

    step_classifications = []
    overall_risk = RiskLevel.SAFE

    for step_name, model, method in steps_def:
        classification = classify_operation(model, method, role=role)
        cascade_warning = CASCADE_WARNINGS.get((model, method))

        step_cls = WorkflowStepClassification(
            step=step_name,
            model=model,
            method=method,
            risk_level=classification.risk_level,
            cascade_warning=cascade_warning,
        )
        step_classifications.append(step_cls)

        if _RISK_ORDER[classification.risk_level] > _RISK_ORDER[overall_risk]:
            overall_risk = classification.risk_level

    # Build human-readable message
    high_steps = [s for s in step_classifications if s.risk_level in (RiskLevel.HIGH, RiskLevel.BLOCKED)]
    warnings = [s.cascade_warning for s in step_classifications if s.cascade_warning]

    message_parts = [
        f"Workflow '{workflow}' contains {len(step_classifications)} steps "
        f"with overall risk level: {overall_risk.value}."
    ]
    if high_steps:
        message_parts.append(f"High-risk steps: {', '.join(s.step for s in high_steps)}.")
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
    """Check if audit logging is enabled (read from env on each call)."""
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

        logger.info("[SAFETY AUDIT] %s", json.dumps(entry))
    except Exception as exc:
        try:
            logger.error("[SAFETY AUDIT ERROR] %s", exc)
        except Exception:
            pass


# ----- Payload Pre-flight Validation (Phase 2) -----


@dataclass(frozen=True)
class PayloadValidationResult:
    ok: bool
    errors: list[str]


# Where the vals payload lives for each method: (positional index, kwargs key).
# The v2 API is named-args-only, so the same payload reaches us either way and
# both spellings must be read — checking only the positional form lets a write
# skip validation by moving its vals into kwargs_json. Same class of bug as
# _estimate_record_count before it learned to read _COUNTED_ARG from kwargs.
_VALS_ARG: dict[str, tuple[int, str]] = {
    "create": (0, "vals_list"),
    "write": (1, "vals"),
    "copy": (1, "default"),
}


def _extract_vals_dict(method: str, args: list, kwargs: dict | None = None) -> dict | None:
    """Pull the vals dict out of an operation's args or kwargs, by method."""
    spec = _VALS_ARG.get(method)
    if spec is None:
        return None  # action_*, button_*, unlink: no vals dict
    position, key = spec
    kwargs = kwargs or {}

    raw: Any = None
    if args and len(args) > position:
        raw = args[position]
    elif key in kwargs:
        raw = kwargs[key]

    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0]  # create([{...}]) — validate the first record only
    return None


def validate_payload_against_schema(
    client,
    model: str,
    method: str,
    args: list | None = None,
    kwargs: dict | None = None,
) -> PayloadValidationResult:
    """Validate that a write payload references only real, writable fields.

    Returns ok=True for non-vals methods (action_*, button_*, unlink) — those
    have no payload to validate.

    Empty fields_get response counts as a failure: a silent connection
    drop must not grant a write token.
    """
    from .constants import COMPACT_FIELD_ATTRIBUTES
    from .utils import get_fields_for_model

    args = args or []
    vals = _extract_vals_dict(method, args, kwargs)
    if vals is None:
        return PayloadValidationResult(ok=True, errors=[])

    # Same attribute subset as the compact schema resources, so a quick-schema
    # read followed by a gated write costs one fields_get, not two.
    fields = get_fields_for_model(client, model, attributes=COMPACT_FIELD_ATTRIBUTES)
    if not fields:
        return PayloadValidationResult(
            ok=False,
            errors=[
                f"Could not load schema for '{model}' (fields_get returned "
                f"empty). Refusing to issue a confirmation token without a "
                f"verified field list."
            ],
        )

    errors: list[str] = []
    for field_name, value in vals.items():
        if field_name == "context":
            continue  # context is a kwargs concern, not a vals field
        spec = fields.get(field_name)
        if spec is None:
            errors.append(
                f"Field '{field_name}' does not exist on model '{model}'. "
                f"Read odoo://model/{model}/quick-schema for the field list."
            )
            continue
        if spec.get("readonly"):
            errors.append(f"Field '{field_name}' is readonly on '{model}' and cannot " f"be written.")

    return PayloadValidationResult(ok=not errors, errors=errors)
