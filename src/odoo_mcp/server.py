"""
MCP Server for Odoo 19+

Provides MCP tools and resources for interacting with Odoo ERP via JSON-2 API.

MCP 2025-11-25 Features:
- Background Tasks (SEP-1686) - Async operations with progress tracking
- Icons (SEP-973) - Visual icons for server and components
- Structured Output Schemas - Typed Pydantic responses
- User Elicitation - Interactive configuration
"""

import asyncio
import functools
import hashlib
import json
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TypeVar

import anyio
from fastmcp import Context
from fastmcp.dependencies import Progress

from . import prompts as _prompts  # noqa: F401 -- import triggers prompt registration
from . import resources as _resources  # noqa: F401 -- import triggers resource registration
from . import skill_prompts as _skill_prompts  # noqa: F401 -- import triggers skill prompt registration
from .app import ODOO_ICON, mcp  # noqa: F401 -- mcp import triggers FastMCP setup
from .constants import (
    _READ_RESOURCE_MAX_CHARS,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    PRIVATE_METHOD_HINTS,
    _merge_context,
    _validate_method,
    _validate_model,
)
from .models import (
    BatchExecuteResponse,
    BatchOperationResult,
    ExecuteMethodResponse,
    ExecuteWorkflowResponse,
    IssueAnalysis,
    WorkflowStepResult,
)
from .odoo_client import get_odoo_client
from .safety import (
    BLOCKED_MODELS,
    RiskLevel,
    SafetyClassification,
    audit_log,
    classify_batch,
    classify_operation,
    classify_workflow,
    is_side_effect_method,
)
from .safety_profile import get_profile
from .user_clients import current_role
from .utils import (
    _get_live_doc,
    _track_model_issue,
    get_error_suggestion,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


async def _run_blocking(fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Run a synchronous OdooClient call in the anyio worker threadpool.

    ``batch_execute`` and ``execute_workflow`` are ``async def`` (they report
    progress), but ``OdooClient`` is built on synchronous ``requests``. Calling
    it inline would block the event loop for the whole round-trip — in HTTP
    multi-user mode a 100-op batch would freeze every session for 100 × RTT.
    Contextvars propagate through ``anyio.to_thread``, so ``get_odoo_client()``
    and the current access token resolve exactly as on the loop thread.
    """
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


# ----- Confirmation Token Store -----
# Stateful nonces that tie a confirmed=True re-call to the original safety classification
# AND the original payload (args/kwargs/operations/params). Prevents an agent from bypassing
# the gate by passing confirmed=true with substituted arguments.

# token → (timestamp, model, method, payload_digest)
_CONFIRMATION_TOKENS: dict[str, tuple[float, str, str, str]] = {}
_CONFIRMATION_LOCK = threading.Lock()
_CONFIRMATION_TTL = 120  # seconds


def _payload_digest(payload: Any) -> str:
    """SHA-256 hex digest of a JSON-serializable payload, with sorted keys for determinism.

    Used to bind a confirmation token to the exact arguments seen at gate-issue time.
    Any change between issue and consume — added IDs, swapped model on a resolve target,
    different operation count or content — produces a different digest, invalidating the token.
    """
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _issue_confirmation_token(model: str, method: str, payload_digest: str) -> str:
    """Issue a short-lived nonce for a pending confirmation."""
    token = secrets.token_urlsafe(16)
    now = time.time()
    with _CONFIRMATION_LOCK:
        # Evict expired tokens
        expired = [k for k, (ts, *_) in _CONFIRMATION_TOKENS.items() if now - ts > _CONFIRMATION_TTL]
        for k in expired:
            del _CONFIRMATION_TOKENS[k]
        _CONFIRMATION_TOKENS[token] = (now, model, method, payload_digest)
    return token


def _validate_confirmation_token(token: str | None, model: str, method: str, payload_digest: str) -> str | None:
    """Validate and consume a confirmation token. Returns error message or None if valid."""
    if not token:
        return "confirmed=true requires a confirmation_token from the safety gate response."
    with _CONFIRMATION_LOCK:
        entry = _CONFIRMATION_TOKENS.pop(token, None)
    if not entry:
        return "Confirmation token is invalid or already used."
    ts, stored_model, stored_method, stored_digest = entry
    if time.time() - ts > _CONFIRMATION_TTL:
        return f"Confirmation token expired (>{_CONFIRMATION_TTL}s). Re-call without confirmed=true to get a new token."
    if stored_model != model or stored_method != method:
        return f"Confirmation token was issued for {stored_model}.{stored_method}, not {model}.{method}."
    if stored_digest != payload_digest:
        return (
            "Confirmation token was issued for a different payload. The arguments must match "
            "exactly between the gate response and the confirmation re-call. Re-call without "
            "confirmed=true to get a new token for the current payload."
        )
    return None


# ----- Shared guards for the write-capable tools -----


def _elapsed_ms(start_time: float) -> float:
    return round((time.time() - start_time) * 1000, 2)


def _read_only_error(subject: str) -> str:
    """Uniform rejection text for the MCP_READ_ONLY / locked-mode kill-switch."""
    return (
        f"read-only mode is active: {subject}. Set MCP_READ_ONLY=false to enable "
        f"writes (or change MCP_SAFETY_MODE away from 'locked' if MCP_READ_ONLY "
        f"is not set explicitly)."
    )


@dataclass(frozen=True)
class _GateDecision:
    """Outcome of the confirmation gate for one gated operation.

    ``outcome`` is ``"issue"`` (first call — hand ``token`` back to the caller),
    ``"reject"`` (confirmed re-call with a bad token — surface ``error``) or
    ``"proceed"`` (token validated and consumed).
    """

    outcome: str
    token: str | None = None
    error: str | None = None


def _confirmation_gate(
    model_key: str,
    method_key: str,
    payload: Any,
    confirmed: bool,
    confirmation_token: str | None,
) -> _GateDecision:
    """Issue a token on the first call; validate and consume it on the confirmed re-call.

    ``payload`` is the exact operation content the token must be bound to —
    post-resolve args/kwargs for ``execute_method``, the operations list for
    ``batch_execute``, the params dict for ``execute_workflow``. Its digest is
    what stops a re-call from substituting arguments after the gate was shown.
    """
    digest = _payload_digest(payload)
    if not confirmed:
        return _GateDecision("issue", token=_issue_confirmation_token(model_key, method_key, digest))
    error = _validate_confirmation_token(confirmation_token, model_key, method_key, digest)
    if error:
        return _GateDecision("reject", error=f"Confirmation rejected: {error}")
    return _GateDecision("proceed")


# ----- execute_method phases -----


def _parse_json_args(args_json: str | None, kwargs_json: str | None) -> tuple[list, dict, str | None]:
    """Decode the JSON-string parameters and merge MCP_DEFAULT_CONTEXT.

    Returns ``(args, kwargs, error)``; ``error`` is set on malformed input.
    """
    args: list = []
    kwargs: dict = {}
    if args_json:
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError as e:
            return [], {}, f"Invalid args_json: {e}"
        if not isinstance(args, list):
            return [], {}, "args_json must be a JSON array"
    if kwargs_json:
        try:
            kwargs = json.loads(kwargs_json)
        except json.JSONDecodeError as e:
            return [], {}, f"Invalid kwargs_json: {e}"
        if not isinstance(kwargs, dict):
            return [], {}, "kwargs_json must be a JSON object"
    merged_ctx = _merge_context(kwargs.get("context"))
    if merged_ctx is not None:
        kwargs = {**kwargs, "context": merged_ctx}
    return args, kwargs, None


def _lookup_resolve_target(
    odoo: Any, field_name: str, spec: Any, start_time: float
) -> tuple[int | None, ExecuteMethodResponse | None]:
    """Resolve one ``resolve_json`` entry to a record id via ``name_search``."""
    target_model = spec.get("model") if isinstance(spec, dict) else None
    search_term = spec.get("search") if isinstance(spec, dict) else None
    if not target_model or not search_term:
        return None, ExecuteMethodResponse(
            success=False, error=f"resolve_json['{field_name}'] requires 'model' and 'search' keys"
        )
    model_err = _validate_model(target_model)
    if model_err:
        return None, ExecuteMethodResponse(success=False, error=f"resolve_json['{field_name}']: {model_err}")
    # Block reads against security-critical models
    if target_model in BLOCKED_MODELS:
        return None, ExecuteMethodResponse(
            success=False, error=f"resolve_json['{field_name}']: model '{target_model}' is blocked for safety."
        )
    # The lookup *and* the unpacking of its result share one boundary: a malformed
    # name_search tuple must surface as a resolve_json failure, not as a generic error.
    try:
        matches = odoo.execute_method(target_model, "name_search", name=search_term, limit=5)
        if not matches:
            return None, ExecuteMethodResponse(
                success=False,
                error=f"resolve_json: No match for '{search_term}' in {target_model}",
                hint=f"Search {target_model} manually to find the correct record",
                execution_time_ms=_elapsed_ms(start_time),
            )
        if len(matches) > 1:
            options = [f"  {m[0]}: {m[1]}" for m in matches[:5]]
            return None, ExecuteMethodResponse(
                success=False,
                error=f"resolve_json: Ambiguous match for '{search_term}' in {target_model} ({len(matches)} results)",
                hint="Multiple matches found:\n" + "\n".join(options) + "\nUse the numeric ID directly instead.",
                execution_time_ms=_elapsed_ms(start_time),
            )
        return matches[0][0], None
    except Exception as e:
        return None, ExecuteMethodResponse(
            success=False,
            error=f"resolve_json: Failed to resolve '{field_name}': {e}",
            execution_time_ms=_elapsed_ms(start_time),
        )


def _inject_resolved_values(method: str, args: list, resolved: dict[str, int]) -> list:
    """Return a copy of ``args`` with the resolved ids merged into the vals dict(s)."""
    if not resolved or not args:
        return args
    if method == "write" and len(args) >= 2 and isinstance(args[1], dict):
        return [args[0], {**args[1], **resolved}, *args[2:]]
    if method == "create":
        if isinstance(args[0], dict):
            return [{**args[0], **resolved}, *args[1:]]
        if isinstance(args[0], list):
            vals_list = [{**vals, **resolved} if isinstance(vals, dict) else vals for vals in args[0]]
            return [vals_list, *args[1:]]
    return args


def _resolve_many2one_names(
    odoo: Any, resolve_json: str, method: str, args: list, start_time: float
) -> tuple[list, ExecuteMethodResponse | None]:
    """Apply ``resolve_json``: name_search every entry and inject the ids into the payload."""
    try:
        resolves = json.loads(resolve_json)
    except json.JSONDecodeError as e:
        return args, ExecuteMethodResponse(success=False, error=f"Invalid resolve_json: {e}")
    if not isinstance(resolves, dict):
        return args, ExecuteMethodResponse(success=False, error="resolve_json must be a JSON object")
    resolved: dict[str, int] = {}
    for field_name, spec in resolves.items():
        record_id, failure = _lookup_resolve_target(odoo, field_name, spec, start_time)
        if failure:
            return args, failure
        resolved[field_name] = record_id  # type: ignore[assignment]
    return _inject_resolved_values(method, args, resolved), None


def _reject_private_method(model: str, method: str, start_time: float) -> ExecuteMethodResponse | None:
    """Refuse ``@api.private`` methods: static hint table first, then the live /doc-bearer/ check."""
    if method in PRIVATE_METHOD_HINTS:
        return ExecuteMethodResponse(
            success=False,
            error=f"Method '{method}' is @api.private and cannot be called via RPC.",
            hint=PRIVATE_METHOD_HINTS[method],
            execution_time_ms=_elapsed_ms(start_time),
        )
    if method.startswith("_"):
        live_doc = _get_live_doc(model)
        if live_doc and method not in live_doc.get("methods", {}):
            return ExecuteMethodResponse(
                success=False,
                error=f"Method '{method}' is not a public method on {model}. It may be @api.private or doesn't exist.",
                hint=f"Use odoo://methods/{model} to see available public methods.",
                execution_time_ms=_elapsed_ms(start_time),
            )
    return None


def _classify_and_gate(
    odoo: Any,
    model: str,
    method: str,
    args: list,
    kwargs: dict,
    confirmed: bool,
    confirmation_token: str | None,
    start_time: float,
) -> ExecuteMethodResponse | None:
    """Run the safety classifier, the payload pre-flight and the confirmation gate.

    Returns the response to send when the call must stop here (blocked, token
    issued, token rejected, payload invalid), or ``None`` when execution may proceed.
    """
    classification = classify_operation(model, method, args, kwargs, role=current_role())

    if classification.risk_level == RiskLevel.BLOCKED:
        audit_log(classification, confirmed=confirmed, executed=False)
        return ExecuteMethodResponse(
            success=False,
            pending_confirmation=True,
            safety=classification,
            error=classification.blocked_reason,
            hint="Use the Odoo web interface instead.",
            execution_time_ms=_elapsed_ms(start_time),
        )

    if classification.requires_confirmation:
        # Payload pre-flight against live fields_get — only when the profile asks
        # for it AND this is a write-shaped call — so a typo'd field fails before
        # a gate round-trip is spent on it.
        if get_profile().validate_payloads and is_side_effect_method(method):
            from .safety import validate_payload_against_schema as _validate_payload

            validation = _validate_payload(odoo, model, method, args=args, kwargs=kwargs)
            if not validation.ok:
                return ExecuteMethodResponse(
                    success=False,
                    error="Payload validation failed:\n  - " + "\n  - ".join(validation.errors),
                    hint=f"Read odoo://model/{model}/quick-schema for the field list.",
                    execution_time_ms=_elapsed_ms(start_time),
                )

        # Args/kwargs are post-resolve_json and post-context-merge, so the digest
        # captures what would actually be sent to Odoo.
        decision = _confirmation_gate(model, method, {"args": args, "kwargs": kwargs}, confirmed, confirmation_token)
        if decision.outcome == "issue":
            audit_log(classification, confirmed=False, executed=False)
            message = classification.reason
            if classification.cascade_warning:
                message += f"\n\nWARNING: {classification.cascade_warning}"
            return ExecuteMethodResponse(
                success=False,
                pending_confirmation=True,
                safety=classification,
                error=message,
                hint=(
                    f"Re-call execute_method with confirmed=true and "
                    f"confirmation_token='{decision.token}' to proceed."
                ),
                execution_time_ms=_elapsed_ms(start_time),
            )
        if decision.outcome == "reject":
            return ExecuteMethodResponse(success=False, error=decision.error, execution_time_ms=_elapsed_ms(start_time))

    if classification.risk_level != RiskLevel.SAFE:
        audit_log(classification, confirmed=confirmed, executed=True)
    return None


def _apply_search_defaults(method: str, args: list, kwargs: dict) -> tuple[list, dict]:
    """Default/cap the search limit and unwrap a double-wrapped ``[[domain]]``."""
    if method in ("search", "search_read"):
        if "limit" not in kwargs:
            kwargs = {**kwargs, "limit": DEFAULT_LIMIT}
            logger.debug("Applied default limit=%d", DEFAULT_LIMIT)
        elif kwargs.get("limit", 0) > MAX_LIMIT:
            kwargs = {**kwargs, "limit": MAX_LIMIT}
            logger.debug("Capped limit to %d", MAX_LIMIT)
    if method in ("search", "search_read", "search_count") and args:
        domain = args[0]
        if isinstance(domain, list) and len(domain) == 1 and isinstance(domain[0], list):
            if domain[0] and isinstance(domain[0][0], list):
                args = [domain[0], *args[1:]]
    return args, kwargs


def _search_read_fallback(
    odoo: Any, model: str, kwargs: dict, error_msg: str, start_time: float
) -> ExecuteMethodResponse:
    """Retry a failed ``search_read`` as ``search`` + ``read`` and record the runtime issue."""
    domain = kwargs.get("domain", [])
    fields = kwargs.get("fields", [])
    context = kwargs.get("context")
    try:
        search_kwargs: dict[str, Any] = {
            "domain": domain,
            "limit": kwargs.get("limit", 100),
            "offset": kwargs.get("offset", 0),
        }
        if kwargs.get("order"):
            search_kwargs["order"] = kwargs["order"]
        if context:
            search_kwargs["context"] = context
        ids = odoo.execute_method(model, "search", **search_kwargs)

        result: Any = []
        if ids:
            read_kwargs: dict[str, Any] = {}
            if fields:
                read_kwargs["fields"] = fields
            if context:
                read_kwargs["context"] = context
            result = odoo.execute_method(model, "read", ids, **read_kwargs)

        analysis = _track_model_issue(model, "search_read", error_msg, domain=domain, fields=fields)
        return ExecuteMethodResponse(
            success=True,
            result=result,
            fallback_used=True,
            issue_analysis=IssueAnalysis(
                category=analysis["category"],
                cause=analysis["cause"],
                domain_patterns=analysis["domain_patterns"],
                problematic_fields=analysis.get("problematic_fields", []),
                suggested_solutions=analysis["solutions"][:2],
                model_specific_advice=analysis.get("model_specific_advice", []),
            ),
            note=f"Fallback search+read used. Cause: {analysis['cause']}",
            execution_time_ms=_elapsed_ms(start_time),
        )
    except Exception as fallback_error:
        return ExecuteMethodResponse(
            success=False,
            error=f"{error_msg}; Fallback also failed: {fallback_error}",
            suggestion=(
                "Both search_read and fallback search+read failed. " "Check odoo://model-limitations for known issues."
            ),
            execution_time_ms=_elapsed_ms(start_time),
        )


_FIELD_ERROR_PATTERNS = ("invalid field", "unknown field", "field_get", "keyerror", "no field", "does not exist")


def _failure_response(model: str, method: str, error_msg: str, start_time: float) -> ExecuteMethodResponse:
    """Wrap an Odoo error with a pattern-matched suggestion and a schema hint."""
    suggestion = get_error_suggestion(error_msg, model, method)
    hint = None
    if any(p in error_msg.lower() for p in _FIELD_ERROR_PATTERNS):
        hint = (
            f"Field name error detected. Read odoo://model/{model}/fields to get exact field names, "
            f"or odoo://model/{model}/schema for full details."
        )
    elif suggestion:
        hint = f"Check odoo://methods/{model} or odoo://module-knowledge for special methods"
    return ExecuteMethodResponse(
        success=False,
        error=error_msg,
        suggestion=suggestion,
        hint=hint,
        execution_time_ms=_elapsed_ms(start_time),
    )


# ----- MCP Tools (execute_method, batch_execute, execute_workflow, configure_odoo, read_resource) -----

# Icon list for tools (reusable)
_tool_icons = [ODOO_ICON] if ODOO_ICON else None


@mcp.tool(
    description="""Execute ANY Odoo method on ANY model.

    This is the universal tool for full Odoo API access.

    BEFORE USING: Read these resources for guidance:
    - odoo://actions/{model} - Discover available actions
    - odoo://methods/{model} - Method signatures
    - odoo://domain-syntax - Domain filter reference
    - odoo://aggregation - read_group guide

    MANDATORY WORKFLOW (no guessing!):
    1. FIRST: Read odoo://model/{model}/quick-schema to get exact field names/types
    2. THEN: Build your query using schema field names
    Never guess field names - introspect schema first to avoid failed requests.

    Common patterns:
    - search_read: kwargs_json='{"domain": [...], "fields": [...], "limit": 100}'
    - create: kwargs_json='{"vals_list": [{"field": "value"}]}'
    - write: args_json='[[ids], {"field": "value"}]'
    - unlink: args_json='[[ids]]'
    - formatted_read_group (v19): kwargs_json='{"domain": [...], "groupby": ["field"], "aggregates": ["field:sum"]}'

    RELATIONAL FIELD WRITES (critical -- wrong syntax = silent failure):
    - Many2one: ALWAYS numeric ID, never name. Use resolve_json to auto-resolve.
    - Many2many: (4, id) link, (3, id) unlink, (6, 0, [ids]) replace all, (5, 0, 0) clear
    - One2many: (0, 0, {vals}) create child, (1, id, {vals}) update, (2, id, 0) delete

    DATE/DATETIME FORMATS (wrong format = 500 or silent failure):
    - date fields: "YYYY-MM-DD" (e.g. "2026-04-10")
    - datetime fields: "YYYY-MM-DD HH:MM:SS" in UTC (e.g. "2026-04-10 14:30:00")

    STATE CHANGES: Never use write() to change state fields directly.
    Use action methods instead (action_confirm, action_set_won, action_post, etc.).
    Read odoo://methods/{model} to find the correct action method.

    Smart limits: Default 100, Max 1000 records

    DISCOVERY RESOURCES (read these before querying):
    - odoo://model/{model}/quick-schema - Compact field names & types (densest form, best for token savings)
    - odoo://model/{model}/fields - Lightweight field list (e.g. odoo://model/res.partner/fields)
    - odoo://model/{model}/workflow - State machine transitions (e.g. odoo://model/sale.order/workflow)
    - odoo://bundle/{models} - Batch quick-schema for N models (e.g. odoo://bundle/res.partner,sale.order)
    - odoo://session-bootstrap - Bootstrap conversation with schemas + workflows
    - odoo://methods/{model} - Available methods (e.g. odoo://methods/crm.lead)
    - odoo://actions/{model} - Discover actions (e.g. odoo://actions/sale.order)
    - odoo://model/{model}/docs - Rich docs with help text
    - odoo://record/{model}/{id} - Read a record
    - odoo://find-model/{concept} - Natural language lookup
    - odoo://tools/{query} - Search operations
    - odoo://docs/{target} - Documentation URLs
    - odoo://module-knowledge/{name} - Module-specific methods

    SAFETY: Dangerous operations return pending_confirmation=true.
    Add confirmed=true to proceed after reviewing the classification.
    """,
    annotations={
        "title": "Execute Odoo Method",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
    icons=_tool_icons,
)
def execute_method(
    ctx: Context,
    model: str,
    method: str,
    args_json: str = None,
    kwargs_json: str = None,
    confirmed: bool = False,
    confirmation_token: str = None,
    resolve_json: str = None,
) -> ExecuteMethodResponse:
    """
    Execute any method on an Odoo model.

    Parameters:
        model: Model name (e.g., 'res.partner')
        method: Method name (e.g., 'search_read', 'create')
        args_json: JSON array of positional arguments
        kwargs_json: JSON object of keyword arguments
        confirmed: Set to true to bypass safety confirmation
        resolve_json: JSON object to auto-resolve Many2one names to IDs.
            Format: '{"field_name": {"model": "target.model", "search": "name to find"}}'
            Resolves via name_search before execution. Errors if 0 or >1 matches.

    Examples:
        Search partners:
            model='res.partner'
            method='search_read'
            args_json='[[["is_company", "=", true]]]'
            kwargs_json='{"fields": ["name", "email"], "limit": 10}'

        Create record:
            model='res.partner'
            method='create'
            args_json='[{"name": "Test Company"}]'

        Write with auto-resolved Many2one:
            model='res.partner'
            method='write'
            args_json='[[1], {"user_id": null}]'
            resolve_json='{"user_id": {"model": "res.users", "search": "Administrator"}}'
    """
    start_time = time.time()

    model_err = _validate_model(model)
    if model_err:
        return ExecuteMethodResponse(success=False, error=model_err, execution_time_ms=0)
    method_err = _validate_method(method)
    if method_err:
        return ExecuteMethodResponse(success=False, error=method_err, execution_time_ms=0)

    # Read-only kill-switch — cheap set lookup, runs before the classifier and any round-trip.
    if get_profile().read_only and is_side_effect_method(method):
        return ExecuteMethodResponse(
            success=False,
            error=_read_only_error(f"'{method}' on '{model}' is a side-effect operation"),
            execution_time_ms=_elapsed_ms(start_time),
        )

    odoo = get_odoo_client()
    # Runs outside the try below on purpose: _parse_json_args reports malformed input
    # as a response and must never raise (json errors are caught inside it and
    # _merge_context swallows its own). Keep that invariant if you touch either.
    args, kwargs, parse_err = _parse_json_args(args_json, kwargs_json)
    if parse_err:
        return ExecuteMethodResponse(success=False, error=parse_err)

    try:
        if resolve_json:
            args, failure = _resolve_many2one_names(odoo, resolve_json, method, args, start_time)
            if failure:
                return failure

        rejected = _reject_private_method(model, method, start_time)
        if rejected:
            return rejected

        gated = _classify_and_gate(odoo, model, method, args, kwargs, confirmed, confirmation_token, start_time)
        if gated:
            return gated

        args, kwargs = _apply_search_defaults(method, args, kwargs)
        result = odoo.execute_method(model, method, *args, **kwargs)
        return ExecuteMethodResponse(success=True, result=result, execution_time_ms=_elapsed_ms(start_time))

    except Exception as e:
        error_msg = str(e)
        if method == "search_read" and ("500" in error_msg or "Internal Server Error" in error_msg):
            return _search_read_fallback(odoo, model, kwargs, error_msg, start_time)
        return _failure_response(model, method, error_msg, start_time)


@mcp.tool(
    description="""Execute multiple Odoo operations in a batch with progress tracking.

    SAFETY: Dangerous operations return pending_confirmation=true.
    Add confirmed=true to proceed after reviewing the safety preview.""",
    annotations={
        "title": "Batch Execute",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
    icons=_tool_icons,
    task=True,  # Enable background task execution with progress
)
async def batch_execute(
    operations: List[Dict[str, Any]],
    atomic: bool = True,
    confirmed: bool = False,
    confirmation_token: str = None,
    progress: Progress = Progress(),
) -> BatchExecuteResponse:
    """
    Execute multiple operations efficiently with progress tracking.

    Parameters:
        operations: List of operations, each with:
            - model: str (required)
            - method: str (required)
            - args_json: str (optional)
            - kwargs_json: str (optional)
        atomic: If True, fail fast on first error
        confirmed: Set to true to bypass safety confirmation
    """
    start_time = time.time()

    # Read-only kill-switch — reject the whole batch if any operation is a
    # side-effect call. Runs before classification.
    profile = get_profile()
    if profile.read_only:
        for op in operations:
            method = op.get("method", "")
            if is_side_effect_method(method):
                return BatchExecuteResponse(
                    success=False,
                    error=_read_only_error(
                        f"batch contains side-effect operation '{method}' on '{op.get('model', '?')}'"
                    ),
                    results=[],
                    total_operations=len(operations),
                    successful_operations=0,
                    failed_operations=0,
                )

    # Get Odoo client directly (works in both sync and background task modes)
    odoo = get_odoo_client()
    results: List[BatchOperationResult] = []
    successful = 0
    failed = 0

    # Set up progress tracking
    await progress.set_total(len(operations))

    # --- Safety Classification for batch ---
    classifications, overall_risk, any_needs_confirmation = classify_batch(operations, role=current_role())

    # BLOCKED operations always refuse
    blocked = [c for c in classifications if c.risk_level == RiskLevel.BLOCKED]
    if blocked:
        for c in blocked:
            audit_log(c, confirmed=confirmed, executed=False)
        elapsed_ms = (time.time() - start_time) * 1000
        blocked_models = ", ".join(set(c.model for c in blocked))
        return BatchExecuteResponse(
            success=False,
            results=[],
            total_operations=len(operations),
            successful_operations=0,
            failed_operations=0,
            error=f"Batch contains blocked operations on: {blocked_models}. Remove them and retry.",
            pending_confirmation=True,
            safety_preview=classifications,
            overall_risk=overall_risk.value,
            execution_time_ms=round(elapsed_ms, 2),
        )

    # Operations needing confirmation
    if any_needs_confirmation:
        # Bind the token to the exact list of operations. Substituting any op (or even
        # a single arg within an op) on the re-call produces a different digest.
        decision = _confirmation_gate("__batch__", "batch", operations, confirmed, confirmation_token)
        if decision.outcome == "issue":
            for c in classifications:
                if c.requires_confirmation:
                    audit_log(c, confirmed=False, executed=False)
            elapsed_ms = (time.time() - start_time) * 1000
            return BatchExecuteResponse(
                success=False,
                results=[],
                total_operations=len(operations),
                successful_operations=0,
                failed_operations=0,
                error=f"Batch contains operations that require confirmation. Review safety_preview and re-call with confirmed=true and confirmation_token='{decision.token}'.",
                pending_confirmation=True,
                safety_preview=classifications,
                overall_risk=overall_risk.value,
                execution_time_ms=round(elapsed_ms, 2),
            )
        if decision.outcome == "reject":
            return BatchExecuteResponse(
                success=False,
                results=[],
                total_operations=len(operations),
                successful_operations=0,
                failed_operations=0,
                error=decision.error,
                execution_time_ms=_elapsed_ms(start_time),
            )

    # Audit non-safe operations that will proceed
    for c in classifications:
        if c.risk_level != RiskLevel.SAFE:
            audit_log(c, confirmed=confirmed, executed=True)

    try:
        for idx, op in enumerate(operations):
            model = op.get("model", "unknown")
            method = op.get("method", "unknown")
            await progress.set_message(f"Operation {idx + 1}/{len(operations)}: {model}.{method}")

            try:
                if not op.get("model") or not op.get("method"):
                    raise ValueError(f"Operation {idx}: 'model' and 'method' required")

                model_err = _validate_model(op["model"])
                if model_err:
                    raise ValueError(f"Operation {idx}: {model_err}")
                method_err = _validate_method(op["method"])
                if method_err:
                    raise ValueError(f"Operation {idx}: {method_err}")

                args_json = op.get("args_json")
                kwargs_json = op.get("kwargs_json")

                args = json.loads(args_json) if args_json else []
                if not isinstance(args, list):
                    raise ValueError(f"Operation {idx}: args_json must be a JSON array")
                kwargs = json.loads(kwargs_json) if kwargs_json else {}
                if not isinstance(kwargs, dict):
                    raise ValueError(f"Operation {idx}: kwargs_json must be a JSON object")

                # Merge default context if configured (no-op when env var unset)
                merged_ctx = _merge_context(kwargs.get("context"))
                if merged_ctx is not None:
                    kwargs["context"] = merged_ctx

                result = await _run_blocking(odoo.execute_method, model, method, *args, **kwargs)
                results.append(BatchOperationResult(operation_index=idx, success=True, result=result))
                successful += 1

            except Exception as e:
                results.append(BatchOperationResult(operation_index=idx, success=False, error=str(e)))
                failed += 1

                if atomic:
                    elapsed_ms = (time.time() - start_time) * 1000
                    return BatchExecuteResponse(
                        success=False,
                        results=results,
                        total_operations=len(operations),
                        successful_operations=successful,
                        failed_operations=failed,
                        error=f"Failed at operation {idx}: {e}",
                        execution_time_ms=round(elapsed_ms, 2),
                    )

            await progress.increment()
            # Yield to let the event loop flush progress notifications.
            # Was sleep(0.01) — that added ~1s of dead wall-time to a 100-op batch
            # without buying anything; sleep(0) is enough to schedule pending sends.
            await asyncio.sleep(0)

        elapsed_ms = (time.time() - start_time) * 1000
        return BatchExecuteResponse(
            success=failed == 0,
            results=results,
            total_operations=len(operations),
            successful_operations=successful,
            failed_operations=failed,
            error=None if failed == 0 else f"{failed} operations failed",
            execution_time_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return BatchExecuteResponse(
            success=False,
            results=results,
            total_operations=len(operations),
            successful_operations=successful,
            failed_operations=failed,
            error=str(e),
            execution_time_ms=round(elapsed_ms, 2),
        )


# ----- User Elicitation Tool -----


@dataclass
class OdooConnectionConfig:
    """Configuration collected from user elicitation."""

    url: str
    database: str
    auth_method: str
    username: str


@mcp.tool(
    description="""Interactive Odoo connection configuration using user elicitation.

    This tool guides users through setting up Odoo connection parameters
    interactively, collecting URL, database, and authentication details.

    Note: This requires an MCP client that supports user elicitation.
    The collected configuration is returned but not automatically applied -
    users should set the corresponding environment variables.
    """,
    annotations={
        "title": "Configure Odoo Connection",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    icons=_tool_icons,
)
async def configure_odoo(ctx: Context) -> Dict[str, Any]:
    """
    Interactive Odoo connection configuration using MCP elicitation.

    Returns:
        Configuration summary with environment variable instructions
    """
    from fastmcp.server.elicitation import AcceptedElicitation, CancelledElicitation, DeclinedElicitation

    results = {
        "success": False,
        "config": {},
        "env_vars": {},
    }

    try:
        # Step 1: Ask for Odoo URL
        url_result = await ctx.elicit(
            message="Enter your Odoo server URL (e.g., https://mycompany.odoo.com):",
            response_type=str,
        )

        match url_result:
            case AcceptedElicitation(data=url):
                results["config"]["url"] = url
            case DeclinedElicitation() | CancelledElicitation():
                results["error"] = "Configuration cancelled by user"
                return results

        # Step 2: Ask for database name
        db_result = await ctx.elicit(
            message="Enter the database name:",
            response_type=str,
        )

        match db_result:
            case AcceptedElicitation(data=database):
                results["config"]["database"] = database
            case DeclinedElicitation() | CancelledElicitation():
                results["error"] = "Configuration cancelled by user"
                return results

        # Step 3: Ask for authentication method
        auth_result = await ctx.elicit(
            message="Select authentication method:",
            response_type=["API Key (Recommended)", "Password"],
        )

        match auth_result:
            case AcceptedElicitation(data=auth_method):
                results["config"]["auth_method"] = "api_key" if "API" in auth_method else "password"
            case DeclinedElicitation() | CancelledElicitation():
                results["error"] = "Configuration cancelled by user"
                return results

        # Step 4: Ask for username
        user_result = await ctx.elicit(
            message="Enter your Odoo username (email):",
            response_type=str,
        )

        match user_result:
            case AcceptedElicitation(data=username):
                results["config"]["username"] = username
            case DeclinedElicitation() | CancelledElicitation():
                results["error"] = "Configuration cancelled by user"
                return results

        # Build environment variables
        results["success"] = True
        results["env_vars"] = {
            "ODOO_URL": results["config"]["url"],
            "ODOO_DB": results["config"]["database"],
            "ODOO_USERNAME": results["config"]["username"],
        }

        if results["config"]["auth_method"] == "api_key":
            results["env_vars"]["ODOO_API_KEY"] = "<your-api-key>"
            results["note"] = "Generate an API key in Odoo: Settings > Users > Preferences > API Keys"
        else:
            results["env_vars"]["ODOO_PASSWORD"] = "<your-password>"
            results["note"] = "Using password authentication. API keys are recommended for production."

        results["instructions"] = "Set these environment variables to configure the Odoo MCP server:\n" + "\n".join(
            f"export {k}='{v}'" for k, v in results["env_vars"].items()
        )

        return results

    except Exception as e:
        if "elicitation is not supported" in str(e).lower():
            return {
                "success": False,
                "error": "User elicitation not supported by this MCP client",
                "alternative": "Set environment variables manually: ODOO_URL, ODOO_DB, ODOO_USERNAME, ODOO_API_KEY",
            }
        results["error"] = str(e)
        return results


@mcp.tool(
    description="""Execute a multi-step workflow in a single call with progress tracking.

    Combines several operations into one call, which costs far fewer tokens than
    driving the same steps with individual execute_method calls.

    Supported workflows:
    - lead_to_won (aliases: crm_workflow, opportunity_won)
      Convert a lead to an opportunity -> mark it won. Requires lead_id.
    - create_and_post_invoice (alias: quick_invoice)
      Create a customer invoice -> post it. Requires partner_id and invoice_lines.

    Any other name returns "Unknown workflow" with the list above — describe the
    steps you need with execute_method or batch_execute instead.

    SAFETY: Workflows with dangerous steps return pending_confirmation=true with a
    single-use confirmation_token. Re-call with confirmed=true AND that token.
    """,
    annotations={
        "title": "Execute Workflow",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
    icons=_tool_icons,
    task=True,  # Enable background task execution with progress
)
async def execute_workflow(
    workflow: str,
    params_json: str = None,
    confirmed: bool = False,
    confirmation_token: str = None,
    progress: Progress = Progress(),
) -> ExecuteWorkflowResponse:
    """
    Execute a multi-step workflow with progress tracking.

    Parameters:
        workflow: Workflow name or description
        params_json: JSON object with workflow parameters
        confirmed: Set to true to bypass safety confirmation

    Returns:
        Results from each step of the workflow
    """
    start_time = time.time()

    # Read-only kill-switch — workflows are by definition multi-step actions.
    if get_profile().read_only:
        return ExecuteWorkflowResponse(
            workflow=workflow,
            success=False,
            error=_read_only_error(f"workflow '{workflow}' is a multi-step action"),
        )

    # Get Odoo client directly (works in both sync and background task modes)
    odoo = get_odoo_client()

    try:
        params = json.loads(params_json) if params_json else {}
    except json.JSONDecodeError as e:
        return ExecuteWorkflowResponse(
            workflow=workflow,
            success=False,
            error=f"Invalid params_json: {e}",
        )

    # --- Safety Classification for workflow ---
    safety_preview = classify_workflow(workflow, params, role=current_role())
    if safety_preview is not None:
        # Bind the token to (workflow_name, params). A different order_id or partner_id
        # on the re-call produces a different digest and is rejected.
        workflow_key = workflow.lower().strip()
        decision = _confirmation_gate("__workflow__", workflow_key, params, confirmed, confirmation_token)
        if decision.outcome == "issue":
            elapsed_ms = (time.time() - start_time) * 1000
            return ExecuteWorkflowResponse(
                workflow=workflow,
                success=False,
                pending_confirmation=True,
                safety_preview=[
                    SafetyClassification(
                        risk_level=step.risk_level,
                        model=step.model,
                        method=step.method,
                        record_count=None,
                        requires_confirmation=step.risk_level in (RiskLevel.HIGH, RiskLevel.BLOCKED),
                        reason=f"Step '{step.step}': {step.risk_level.value} risk",
                        cascade_warning=step.cascade_warning,
                    )
                    for step in safety_preview.steps
                ],
                overall_risk=safety_preview.overall_risk.value,
                error=safety_preview.message,
                tip=(
                    f"Re-call execute_workflow with confirmed=true and "
                    f"confirmation_token='{decision.token}' to proceed."
                ),
                execution_time_ms=round(elapsed_ms, 2),
            )
        if decision.outcome == "reject":
            return ExecuteWorkflowResponse(
                workflow=workflow,
                success=False,
                error=decision.error,
                execution_time_ms=_elapsed_ms(start_time),
            )

    workflow_lower = workflow.lower().strip()
    steps: List[WorkflowStepResult] = []

    try:
        # ----- Lead to Won Workflow -----
        if workflow_lower in ["lead_to_won", "crm_workflow", "opportunity_won"]:
            lead_id = params.get("lead_id")

            if not lead_id:
                return ExecuteWorkflowResponse(
                    workflow=workflow,
                    success=False,
                    error="lead_id required for lead_to_won workflow",
                )

            # 2 steps: convert, mark won
            await progress.set_total(2)

            # Step 1: Convert to opportunity (if still a lead)
            await progress.set_message("Converting lead to opportunity...")
            try:
                lead = await _run_blocking(
                    odoo.search_read, "crm.lead", [["id", "=", lead_id]], fields=["type"], limit=1
                )
                if lead and lead[0].get("type") == "lead":
                    await _run_blocking(
                        odoo.execute_method,
                        "crm.lead",
                        "convert_opportunity",
                        [lead_id],
                        partner_id=params.get("partner_id", False),
                    )
                    steps.append(WorkflowStepResult(step="convert_to_opportunity", success=True))
                else:
                    steps.append(
                        WorkflowStepResult(
                            step="convert_to_opportunity", success=True, skipped=True, reason="Already an opportunity"
                        )
                    )
            except Exception as e:
                steps.append(WorkflowStepResult(step="convert_to_opportunity", success=False, error=str(e)))
            await progress.increment()

            # Step 2: Mark as won
            await progress.set_message("Marking opportunity as won...")
            try:
                await _run_blocking(odoo.execute_method, "crm.lead", "action_set_won", [lead_id])
                steps.append(WorkflowStepResult(step="mark_won", success=True))
            except Exception as e:
                steps.append(WorkflowStepResult(step="mark_won", success=False, error=str(e)))
            await progress.increment()

            elapsed_ms = (time.time() - start_time) * 1000
            return ExecuteWorkflowResponse(
                workflow=workflow,
                success=all(s.success or s.skipped for s in steps),
                steps=steps,
                execution_time_ms=round(elapsed_ms, 2),
            )

        # ----- Create and Post Invoice Workflow -----
        elif workflow_lower in ["create_and_post_invoice", "quick_invoice"]:
            partner_id = params.get("partner_id")
            lines = params.get("lines", [])

            if not partner_id:
                return ExecuteWorkflowResponse(
                    workflow=workflow,
                    success=False,
                    error="partner_id required",
                )
            if not lines:
                return ExecuteWorkflowResponse(
                    workflow=workflow,
                    success=False,
                    error="lines required (list of {product_id, quantity, price_unit})",
                )

            # 2 steps: create, post
            await progress.set_total(2)

            # Build invoice lines
            invoice_lines = []
            for line in lines:
                invoice_lines.append(
                    (
                        0,
                        0,
                        {
                            "product_id": line.get("product_id"),
                            "quantity": line.get("quantity", 1),
                            "price_unit": line.get("price_unit"),
                            "name": line.get("name", "Product"),
                        },
                    )
                )

            # Step 1: Create invoice
            await progress.set_message("Creating invoice...")
            invoice_id = None
            try:
                invoice_vals = {
                    "move_type": "out_invoice",
                    "partner_id": partner_id,
                    "invoice_line_ids": invoice_lines,
                }
                invoice_id = await _run_blocking(odoo.execute_method, "account.move", "create", [invoice_vals])
                steps.append(WorkflowStepResult(step="create_invoice", success=True, result={"invoice_id": invoice_id}))
            except Exception as e:
                steps.append(WorkflowStepResult(step="create_invoice", success=False, error=str(e)))
                elapsed_ms = (time.time() - start_time) * 1000
                return ExecuteWorkflowResponse(
                    workflow=workflow,
                    success=False,
                    steps=steps,
                    execution_time_ms=round(elapsed_ms, 2),
                )
            await progress.increment()

            # Step 2: Post invoice
            await progress.set_message("Posting invoice...")
            if params.get("post", True):
                try:
                    await _run_blocking(odoo.execute_method, "account.move", "action_post", [invoice_id])
                    steps.append(WorkflowStepResult(step="post_invoice", success=True))
                except Exception as e:
                    steps.append(WorkflowStepResult(step="post_invoice", success=False, error=str(e)))
            await progress.increment()

            elapsed_ms = (time.time() - start_time) * 1000
            return ExecuteWorkflowResponse(
                workflow=workflow,
                success=all(s.success or s.skipped for s in steps),
                steps=steps,
                invoice_id=invoice_id,
                execution_time_ms=round(elapsed_ms, 2),
            )

        # ----- Unknown workflow -----
        else:
            return ExecuteWorkflowResponse(
                workflow=workflow,
                success=False,
                error=f"Unknown workflow: {workflow}",
                available_workflows=[
                    "lead_to_won - Convert lead and mark as won",
                    "create_and_post_invoice - Create and post a customer invoice",
                ],
                tip="Read odoo://tools/{query} to find available operations",
            )

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return ExecuteWorkflowResponse(
            workflow=workflow,
            success=False,
            steps=steps,
            error=str(e),
            execution_time_ms=round(elapsed_ms, 2),
        )


# ----- URI Routing for read_resource tool -----
# Maps odoo:// URIs to their handler functions. Patterns are compiled once at
# import time (re.Pattern objects) so read_resource doesn't depend on Python's
# implicit re._MAXCACHE for its dispatch path.
# More specific patterns (e.g. /schema, /fields, /docs) MUST come before generic /model/{name}.

_RESOURCE_ROUTES: list[tuple[re.Pattern[str], Any, list[str]]] = [
    (re.compile(pattern), handler, param_names)
    for pattern, handler, param_names in [
        (r"^odoo://models$", _resources.get_models, []),
        (r"^odoo://session-bootstrap$", _resources.get_session_bootstrap, []),
        (r"^odoo://server-status$", _resources._server_status_payload, []),
        (r"^odoo://bundle/(.+)$", _resources.get_bundle, ["models_csv"]),
        (r"^odoo://model/([^/]+)/quick-schema$", _resources.get_model_quick_schema, ["model_name"]),
        (r"^odoo://model/([^/]+)/workflow$", _resources.get_model_workflow, ["model_name"]),
        (r"^odoo://model/([^/]+)/schema$", _resources.get_model_schema, ["model_name"]),
        (r"^odoo://model/([^/]+)/fields$", _resources.get_model_fields_light, ["model_name"]),
        (r"^odoo://model/([^/]+)/docs$", _resources.get_model_docs, ["model_name"]),
        (r"^odoo://model/([^/]+)$", _resources.get_model_info, ["model_name"]),
        (r"^odoo://record/([^/]+)/(\d+)$", _resources.get_record, ["model_name", "record_id"]),
        (r"^odoo://methods/([^/]+)$", _resources.get_methods, ["model_name"]),
        (r"^odoo://find-model/(.+)$", _resources.find_model_resource, ["concept"]),
        (r"^odoo://actions/([^/]+)$", _resources.discover_actions_resource, ["model"]),
        (r"^odoo://tools/(.+)$", _resources.search_tools_resource, ["query"]),
        (r"^odoo://docs/(.+)$", _resources.get_documentation_urls, ["target"]),
        (r"^odoo://module-knowledge/(.+)$", _resources.get_module_knowledge_by_name, ["module_name"]),
        (r"^odoo://module-knowledge$", _resources.get_module_knowledge, []),
        (r"^odoo://concepts$", _resources.get_concept_mappings, []),
        (r"^odoo://templates$", _resources.get_resource_templates, []),
        (r"^odoo://workflows$", _resources.get_workflows, []),
        (r"^odoo://server/info$", _resources.get_server_info, []),
        (r"^odoo://domain-syntax$", _resources.get_domain_syntax, []),
        (r"^odoo://model-limitations$", _resources.get_model_limitations, []),
        (r"^odoo://pagination$", _resources.get_pagination_guide, []),
        (r"^odoo://hierarchical$", _resources.get_hierarchical_guide, []),
        (r"^odoo://aggregation$", _resources.get_aggregation_guide, []),
        (r"^odoo://tool-registry$", _resources.get_tool_registry, []),
    ]
]


@mcp.tool(
    description="""Read any odoo:// resource by URI. Use this for schema discovery, method lookup, guides, etc.

    IMPORTANT: Only fetch resources relevant to the user's current task. Do NOT explore multiple resources just to see what's available.

    Recommended workflow:
    1. odoo://find-model/{concept} - Find the right model name
    2. odoo://model/{model}/quick-schema - Get field names and types (ultra-compact, best for tokens)
    3. odoo://methods/{model} - Check available methods if needed
    Then use execute_method() to query.

    Batch operations:
    - odoo://bundle/{models} - Quick-schema for N models in one call (e.g. odoo://bundle/res.partner,sale.order)
    - odoo://session-bootstrap - Bootstrap conversation with schemas + workflows for common models

    Other useful resources:
    - odoo://model/{model}/workflow - State machine transitions
    - odoo://model/{model}/fields - Lightweight field list (larger than quick-schema, includes labels)
    - odoo://domain-syntax - Domain filter reference
    - odoo://aggregation - Aggregation/groupby guide
    - odoo://templates - List all available resource URIs
    """,
    annotations={
        "title": "Read Resource",
        "readOnlyHint": True,
        "openWorldHint": False,
    },
    icons=_tool_icons,
)
def read_resource(uri: str, max_chars: int = _READ_RESOURCE_MAX_CHARS) -> str:
    """Read an Odoo MCP resource by URI.

    Parameters:
        uri: Resource URI (e.g. 'odoo://model/res.partner/fields')
        max_chars: Max output length in characters (default: 15000). Set to 0 for unlimited.
    """
    if not uri.startswith("odoo://"):
        return json.dumps({"error": "Invalid URI: must start with odoo://", "uri": uri})

    for pattern, handler, param_names in _RESOURCE_ROUTES:
        match = pattern.match(uri)
        if match:
            args = dict(zip(param_names, match.groups()))
            result = handler(**args)
            if max_chars and len(result) > max_chars:
                truncated = result[:max_chars]
                warning = json.dumps(
                    {
                        "_truncated": True,
                        "_total_chars": len(result),
                        "_returned_chars": max_chars,
                        "_hint": f"Output truncated from {len(result):,} to {max_chars:,} chars. "
                        f"Use max_chars=0 for full output, or use narrower queries "
                        f"(e.g. odoo://model/{{model}}/fields instead of /schema).",
                    }
                )
                return truncated + "\n\n" + warning
            return result

    return json.dumps(
        {
            "error": f"Unknown resource URI: {uri}",
            "hint": "Use odoo://templates to list all available resource URIs",
            "examples": [
                "odoo://model/res.partner/schema",
                "odoo://model/sale.order/fields",
                "odoo://methods/res.partner",
                "odoo://find-model/invoice",
                "odoo://domain-syntax",
            ],
        },
        indent=2,
    )
