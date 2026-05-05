# Safety locked-mode + write-path hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `locked` value to `MCP_SAFETY_MODE` that activates four colleague-friendly safety defaults (read-only writes, side-effect allowlist, localhost bind, payload pre-flight), each independently overridable, plus a posture-introspection resource.

**Architecture:** A new `safety_profile.py` module resolves `MCP_SAFETY_MODE` plus four override env vars (`MCP_READ_ONLY`, `MCP_WRITE_ALLOWLIST`, `MCP_HOST`, `MCP_VALIDATE_PAYLOADS`) into a `ResolvedProfile` value object. Downstream modules read the resolved profile via `get_profile()`, which re-reads env on each call so tests stay simple. A new `is_side_effect_method()` predicate in `safety.py` is the single source of truth for "is this call a write?" used by the read-only guard, the allowlist gate, and the payload pre-flight.

**Tech Stack:** Python 3.10+, pytest, FastMCP 3.2.0+, existing `odoo_mcp` modules.

**Spec:** `docs/superpowers/specs/2026-05-05-safety-locked-mode-design.md`

**Branch:** `feat/safety-locked-mode` (already created).

---

## Task 1: Side-effect predicate

**Files:**
- Modify: `src/odoo_mcp/safety.py` (append after `CASCADE_WARNINGS`, before `# ----- Pydantic Models -----`)
- Create: `tests/test_side_effect_predicate.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_side_effect_predicate.py`:

```python
"""Tests for is_side_effect_method() predicate."""
import pytest

from odoo_mcp.safety import is_side_effect_method


@pytest.mark.parametrize("method", [
    "create", "write", "unlink", "copy",
    "action_archive", "action_unarchive", "action_confirm", "action_post",
    "button_validate", "button_confirm", "button_cancel",
])
def test_known_side_effect_methods(method: str):
    assert is_side_effect_method(method) is True


@pytest.mark.parametrize("method", [
    "search_read", "read", "search", "search_count",
    "fields_get", "name_search", "default_get",
    "has_access", "name_get",
])
def test_safe_methods_are_not_side_effects(method: str):
    assert is_side_effect_method(method) is False


@pytest.mark.parametrize("method", [
    "action_my_custom_workflow",
    "button_do_something",
    "_action_private_hook",
])
def test_action_button_patterns(method: str):
    assert is_side_effect_method(method) is True


def test_empty_method_is_not_side_effect():
    assert is_side_effect_method("") is False


def test_unknown_read_like_method_is_not_side_effect():
    # A method we've never heard of that doesn't match any side-effect pattern.
    assert is_side_effect_method("get_widget_count") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_side_effect_predicate.py -v`
Expected: FAIL with `ImportError: cannot import name 'is_side_effect_method' from 'odoo_mcp.safety'`

- [ ] **Step 3: Add the predicate to `safety.py`**

Append after `CASCADE_WARNINGS` (around line 103) and before the `# ----- Pydantic Models -----` section:

```python
# ----- Side-Effect Method Predicate -----

# Methods whose names are explicitly side-effects regardless of pattern.
_LITERAL_SIDE_EFFECT_METHODS = frozenset({
    "create", "write", "unlink", "copy",
    "action_archive", "action_unarchive",
})

# Method-name prefixes that always indicate side effects.
_SIDE_EFFECT_PREFIXES: tuple[str, ...] = (
    "action_", "button_", "_action_",
)


def is_side_effect_method(method: str) -> bool:
    """Return True if calling this method should be treated as a side effect.

    Single source of truth for the read-only guard, the write allowlist, and
    the payload pre-flight. Cheap pattern match — does NOT call the classifier.

    Side-effect methods include:
      * Literal CRUD names (create, write, unlink, copy, action_archive, ...)
      * Anything matching action_*, button_*, _action_*

    SAFE methods (search_read, read, fields_get, ...) and unknown read-like
    methods return False.
    """
    if not method:
        return False
    if method in _LITERAL_SIDE_EFFECT_METHODS:
        return True
    return any(method.startswith(p) for p in _SIDE_EFFECT_PREFIXES)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_side_effect_predicate.py -v`
Expected: PASS, all 22+ test cases green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_side_effect_predicate.py src/odoo_mcp/safety.py
git commit -m "feat(safety): Add is_side_effect_method() predicate

Single source of truth for side-effect detection used by the upcoming
read-only guard, write allowlist, and payload pre-flight. Cheap pattern
match — does not invoke the full classifier."
```

---

## Task 2: ResolvedProfile + resolver

**Files:**
- Create: `src/odoo_mcp/safety_profile.py`
- Create: `tests/test_safety_profile.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_safety_profile.py`:

```python
"""Tests for safety profile resolver."""
import pytest

from odoo_mcp.safety_profile import (
    ResolvedProfile,
    SafetyMode,
    resolve,
)


def test_default_mode_is_strict():
    profile = resolve({})
    assert profile.safety_mode is SafetyMode.STRICT
    assert profile.read_only is False
    assert profile.write_allowlist == frozenset()
    assert profile.write_allowlist_enforced is False
    assert profile.host_default == "0.0.0.0"
    assert profile.validate_payloads is False


def test_permissive_mode():
    profile = resolve({"MCP_SAFETY_MODE": "permissive"})
    assert profile.safety_mode is SafetyMode.PERMISSIVE
    assert profile.read_only is False
    assert profile.write_allowlist_enforced is False
    assert profile.host_default == "0.0.0.0"
    assert profile.validate_payloads is False


def test_locked_mode_defaults():
    profile = resolve({"MCP_SAFETY_MODE": "locked"})
    assert profile.safety_mode is SafetyMode.LOCKED
    assert profile.read_only is True
    assert profile.write_allowlist_enforced is True
    assert profile.host_default == "127.0.0.1"
    assert profile.validate_payloads is True


def test_unknown_mode_falls_back_to_strict():
    # Garbage in env should not crash startup; fall back to safest non-locked.
    profile = resolve({"MCP_SAFETY_MODE": "garbage"})
    assert profile.safety_mode is SafetyMode.STRICT


def test_read_only_override_beats_strict_default():
    profile = resolve({"MCP_SAFETY_MODE": "strict", "MCP_READ_ONLY": "true"})
    assert profile.safety_mode is SafetyMode.STRICT
    assert profile.read_only is True


def test_read_only_override_beats_locked_default():
    # locked has read_only=true by default; explicit false overrides.
    profile = resolve({"MCP_SAFETY_MODE": "locked", "MCP_READ_ONLY": "false"})
    assert profile.safety_mode is SafetyMode.LOCKED
    assert profile.read_only is False


def test_allowlist_explicit_under_strict_enables_enforcement():
    profile = resolve({
        "MCP_SAFETY_MODE": "strict",
        "MCP_WRITE_ALLOWLIST": "sale.order.action_confirm,res.partner.message_post",
    })
    assert profile.write_allowlist_enforced is True
    assert "sale.order.action_confirm" in profile.write_allowlist
    assert "res.partner.message_post" in profile.write_allowlist


def test_allowlist_wildcard_normalised():
    profile = resolve({"MCP_WRITE_ALLOWLIST": "sale.order.*,product.product.write"})
    assert "sale.order.*" in profile.write_allowlist
    assert "product.product.write" in profile.write_allowlist


def test_allowlist_whitespace_tolerant():
    profile = resolve({
        "MCP_WRITE_ALLOWLIST": "  sale.order.action_confirm , product.product.write ",
    })
    assert profile.write_allowlist == frozenset({
        "sale.order.action_confirm",
        "product.product.write",
    })


def test_host_default_under_locked_is_localhost():
    profile = resolve({"MCP_SAFETY_MODE": "locked"})
    assert profile.host_default == "127.0.0.1"


def test_host_explicit_overrides_locked_default():
    profile = resolve({"MCP_SAFETY_MODE": "locked", "MCP_HOST": "0.0.0.0"})
    # Explicit MCP_HOST wins; the resolver only sets the *default* under locked.
    # The bootstrap reads MCP_HOST itself; this test confirms the resolver
    # records the explicit value so the banner / status resource report it.
    assert profile.host == "0.0.0.0"


def test_host_unset_resolves_to_default_per_mode():
    locked = resolve({"MCP_SAFETY_MODE": "locked"})
    assert locked.host == "127.0.0.1"
    strict = resolve({"MCP_SAFETY_MODE": "strict"})
    assert strict.host == "0.0.0.0"


def test_validate_payloads_explicit_enable_under_strict():
    profile = resolve({"MCP_SAFETY_MODE": "strict", "MCP_VALIDATE_PAYLOADS": "true"})
    assert profile.validate_payloads is True


def test_posture_open_only_when_loose():
    """posture_open is true under permissive + 0.0.0.0 + no allowlist."""
    open_ = resolve({"MCP_SAFETY_MODE": "permissive", "MCP_HOST": "0.0.0.0"})
    assert open_.posture_open is True

    locked = resolve({"MCP_SAFETY_MODE": "locked"})
    assert locked.posture_open is False

    permissive_with_allowlist = resolve({
        "MCP_SAFETY_MODE": "permissive",
        "MCP_HOST": "0.0.0.0",
        "MCP_WRITE_ALLOWLIST": "res.partner.message_post",
    })
    assert permissive_with_allowlist.posture_open is False


def test_warnings_flag_locked_with_remote_bind():
    profile = resolve({"MCP_SAFETY_MODE": "locked", "MCP_HOST": "0.0.0.0"})
    assert any("0.0.0.0" in w for w in profile.warnings)


def test_warnings_empty_under_default_strict():
    profile = resolve({})
    assert profile.warnings == ()


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ("", False), ("garbage", False),
])
def test_bool_env_parsing(raw: str, expected: bool):
    profile = resolve({"MCP_READ_ONLY": raw})
    assert profile.read_only is expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_safety_profile.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'odoo_mcp.safety_profile'`

- [ ] **Step 3: Create `safety_profile.py`**

Create `src/odoo_mcp/safety_profile.py`:

```python
"""Safety profile resolution for the Odoo MCP server.

Reads MCP_SAFETY_MODE (umbrella) plus four override env vars and produces a
ResolvedProfile value object. Hybrid umbrella + per-flag override pattern:
the mode picks sensible defaults; explicit individual env vars override them.

Environment variables:
    MCP_SAFETY_MODE: 'permissive' | 'strict' (default) | 'locked'
    MCP_READ_ONLY: 'true' | 'false' — global write kill-switch
    MCP_WRITE_ALLOWLIST: comma-separated 'model.method' entries
    MCP_HOST: host to bind for HTTP transport
    MCP_VALIDATE_PAYLOADS: 'true' | 'false' — Phase 2 fields_get pre-flight
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

logger = logging.getLogger(__name__)


class SafetyMode(str, Enum):
    PERMISSIVE = "permissive"
    STRICT = "strict"
    LOCKED = "locked"


_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off", ""})


def _parse_bool(raw: str | None, default: bool) -> bool:
    """Parse a truthy/falsy env string. Garbage falls back to default."""
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    logger.warning("Could not parse boolean env value %r, using default %r", raw, default)
    return default


def _parse_allowlist(raw: str | None) -> frozenset[str]:
    """Parse a comma-separated 'model.method' list into a frozenset."""
    if not raw:
        return frozenset()
    return frozenset(
        entry.strip()
        for entry in raw.split(",")
        if entry.strip()
    )


def _parse_mode(raw: str | None) -> SafetyMode:
    if not raw:
        return SafetyMode.STRICT
    lowered = raw.strip().lower()
    try:
        return SafetyMode(lowered)
    except ValueError:
        logger.warning(
            "Unknown MCP_SAFETY_MODE=%r, falling back to 'strict'", raw,
        )
        return SafetyMode.STRICT


@dataclass(frozen=True)
class ResolvedProfile:
    """Fully-resolved safety profile read once from env at startup (and on
    each call from get_profile() so tests can monkeypatch env)."""

    safety_mode: SafetyMode
    read_only: bool
    write_allowlist: frozenset[str]
    write_allowlist_enforced: bool
    host: str
    host_default: str
    validate_payloads: bool
    posture_open: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)


def resolve(env: Mapping[str, str]) -> ResolvedProfile:
    """Resolve a ResolvedProfile from a mapping of env vars.

    Pure function: takes a mapping (typically os.environ), returns a frozen
    dataclass. No side effects, no global state.
    """
    mode = _parse_mode(env.get("MCP_SAFETY_MODE"))

    # Defaults derived from mode.
    if mode is SafetyMode.LOCKED:
        default_read_only = True
        default_host = "127.0.0.1"
        default_validate = True
        default_allowlist_enforced = True
    else:
        default_read_only = False
        default_host = "0.0.0.0"
        default_validate = False
        default_allowlist_enforced = False

    # Per-flag overrides.
    read_only = _parse_bool(env.get("MCP_READ_ONLY"), default_read_only)
    validate_payloads = _parse_bool(
        env.get("MCP_VALIDATE_PAYLOADS"), default_validate,
    )
    raw_host = env.get("MCP_HOST")
    host = raw_host if raw_host else default_host

    raw_allowlist = env.get("MCP_WRITE_ALLOWLIST")
    write_allowlist = _parse_allowlist(raw_allowlist)
    # Allowlist enforcement: locked mode always enforces. Explicit allowlist
    # under any mode also enforces (the operator opted in).
    write_allowlist_enforced = (
        default_allowlist_enforced or raw_allowlist is not None
    )

    # posture_open: maximally-loose configuration.
    posture_open = (
        mode is SafetyMode.PERMISSIVE
        and host == "0.0.0.0"
        and not write_allowlist_enforced
        and not read_only
    )

    # Warnings: surface foot-gun combinations explicitly.
    warnings: list[str] = []
    if mode is SafetyMode.LOCKED and host == "0.0.0.0":
        warnings.append(
            "MCP_SAFETY_MODE=locked but MCP_HOST=0.0.0.0 — "
            "remote bind enabled despite locked profile."
        )
    if read_only and write_allowlist:
        warnings.append(
            "MCP_READ_ONLY=true makes MCP_WRITE_ALLOWLIST redundant — "
            "no writes will be permitted regardless of allowlist."
        )

    return ResolvedProfile(
        safety_mode=mode,
        read_only=read_only,
        write_allowlist=write_allowlist,
        write_allowlist_enforced=write_allowlist_enforced,
        host=host,
        host_default=default_host,
        validate_payloads=validate_payloads,
        posture_open=posture_open,
        warnings=tuple(warnings),
    )


def get_profile() -> ResolvedProfile:
    """Resolve the profile from os.environ at call time.

    Re-reads env on each call so tests can monkeypatch env vars and see the
    new profile immediately, matching the existing get_default_context() and
    _get_safety_mode() pattern in the codebase.
    """
    import os
    return resolve(os.environ)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_safety_profile.py -v`
Expected: PASS, all 18+ test cases green.

- [ ] **Step 5: Commit**

```bash
git add src/odoo_mcp/safety_profile.py tests/test_safety_profile.py
git commit -m "feat(safety): Add ResolvedProfile + resolve() with locked mode

Hybrid umbrella + per-flag override resolver. MCP_SAFETY_MODE picks
sensible defaults; MCP_READ_ONLY, MCP_WRITE_ALLOWLIST, MCP_HOST, and
MCP_VALIDATE_PAYLOADS each override individually. Pure function over
an env mapping; get_profile() re-reads os.environ each call to match
the existing testability pattern."
```

---

## Task 3: Read-only enforcement in `execute_method`

**Files:**
- Modify: `src/odoo_mcp/server.py:236` (right after the `_validate_method` check, before `get_odoo_client`)
- Create: `tests/test_read_only_guard.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_read_only_guard.py`:

```python
"""Tests for MCP_READ_ONLY enforcement in execute_method."""
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_ctx():
    return MagicMock()


def _import_execute_method():
    from odoo_mcp.server import execute_method
    return execute_method


def test_read_only_blocks_write(monkeypatch, mock_ctx):
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    execute_method = _import_execute_method()

    response = execute_method(
        ctx=mock_ctx,
        model="res.partner",
        method="write",
        args_json='[[1], {"name": "X"}]',
    )

    assert response.success is False
    assert "read-only" in response.error.lower()
    assert "MCP_READ_ONLY" in response.error


def test_read_only_blocks_create(monkeypatch, mock_ctx):
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    execute_method = _import_execute_method()

    response = execute_method(
        ctx=mock_ctx,
        model="res.partner",
        method="create",
        args_json='[{"name": "Test"}]',
    )

    assert response.success is False
    assert "read-only" in response.error.lower()


def test_read_only_blocks_action_method(monkeypatch, mock_ctx):
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    execute_method = _import_execute_method()

    response = execute_method(
        ctx=mock_ctx,
        model="sale.order",
        method="action_confirm",
        args_json='[[1]]',
    )

    assert response.success is False
    assert "read-only" in response.error.lower()


def test_read_only_allows_safe_method(monkeypatch, mock_ctx):
    """Read-only must NOT short-circuit safe reads — they should reach the
    Odoo client. We can't easily run a real call here, but we can check that
    the read-only guard does not produce a 'read-only mode is active' error
    for a safe method even when MCP_READ_ONLY=true."""
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    # Stop the call from actually contacting Odoo by failing model validation
    # with an obviously-bad model name.
    execute_method = _import_execute_method()

    response = execute_method(
        ctx=mock_ctx,
        model="not a model",   # Will fail _validate_model regex
        method="search_read",
    )

    # The error should be about model validation, NOT read-only mode.
    assert response.success is False
    assert "read-only" not in response.error.lower()


def test_read_only_off_does_not_block(monkeypatch, mock_ctx):
    """When MCP_READ_ONLY is unset/false, the guard must not fire even for
    write methods. We trigger model validation failure to short-circuit
    before the actual Odoo call."""
    monkeypatch.delenv("MCP_READ_ONLY", raising=False)
    execute_method = _import_execute_method()

    response = execute_method(
        ctx=mock_ctx,
        model="not a model",
        method="write",
    )

    assert response.success is False
    assert "read-only" not in response.error.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_read_only_guard.py -v`
Expected: FAIL — at minimum the read-only-true cases should return success=True or different errors because the guard does not exist yet.

- [ ] **Step 3: Add the guard to `execute_method`**

Open `src/odoo_mcp/server.py`. Find the section that ends with `method_err = _validate_method(method)` and the early-return on validation failure (around lines 243-245). Immediately after that block, before `odoo = get_odoo_client()`, insert:

```python
    # Read-only kill-switch (MCP_READ_ONLY / MCP_SAFETY_MODE=locked).
    # Runs before the classifier — cheap pattern match, no Odoo round-trip.
    from .safety import is_side_effect_method
    from .safety_profile import get_profile

    profile = get_profile()
    if profile.read_only and is_side_effect_method(method):
        return ExecuteMethodResponse(
            success=False,
            error=(
                f"read-only mode is active: '{method}' on '{model}' is a "
                f"side-effect operation (set MCP_READ_ONLY=false or "
                f"MCP_SAFETY_MODE=strict to enable writes)."
            ),
            execution_time_ms=int((time.time() - start_time) * 1000),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_read_only_guard.py -v`
Expected: PASS, all 5 cases.

- [ ] **Step 5: Run full unit test suite to catch regressions**

Run: `pytest tests/test_safety.py tests/test_token_gate.py tests/test_side_effect_predicate.py tests/test_safety_profile.py tests/test_read_only_guard.py -v`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/odoo_mcp/server.py tests/test_read_only_guard.py
git commit -m "feat(safety): Add MCP_READ_ONLY guard to execute_method

Side-effect methods short-circuit with a clear error when read-only
mode is active. Runs before the classifier — no Odoo round-trip for
calls that would be blocked anyway. Triggered automatically under
MCP_SAFETY_MODE=locked, also available as standalone MCP_READ_ONLY=true."
```

---

## Task 4: Read-only enforcement in `batch_execute` and `execute_workflow`

**Files:**
- Modify: `src/odoo_mcp/server.py` — find `batch_execute` (line 521) and `execute_workflow` (line 724)
- Modify: `tests/test_read_only_guard.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_read_only_guard.py`:

```python
def test_read_only_blocks_batch_with_writes(monkeypatch, mock_ctx):
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    from odoo_mcp.server import batch_execute

    response = batch_execute(
        ctx=mock_ctx,
        operations_json='[{"model": "res.partner", "method": "write", "args": [[1], {"name": "X"}]}]',
    )

    assert response.success is False
    assert "read-only" in response.error.lower()


def test_read_only_allows_batch_of_reads(monkeypatch, mock_ctx):
    """A batch where every operation is SAFE should not be rejected by the
    read-only guard. We use an invalid model to short-circuit before Odoo."""
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    from odoo_mcp.server import batch_execute

    response = batch_execute(
        ctx=mock_ctx,
        operations_json='[{"model": "not a model", "method": "search_read"}]',
    )

    # The error must not be about read-only — model validation should fire first.
    assert response.success is False
    assert "read-only" not in (response.error or "").lower()


def test_read_only_blocks_workflow_with_writes(monkeypatch, mock_ctx):
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    from odoo_mcp.server import execute_workflow

    response = execute_workflow(
        ctx=mock_ctx,
        workflow_name="quote_to_cash",
        params_json='{"partner_id": 1, "product_id": 1, "quantity": 1}',
    )

    assert response.success is False
    assert "read-only" in response.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_read_only_guard.py -v -k "batch or workflow"`
Expected: FAIL — guard not yet wired into `batch_execute` and `execute_workflow`.

- [ ] **Step 3: Add the guard to `batch_execute`**

Open `src/odoo_mcp/server.py` and find the start of the `batch_execute` function body (after the `def batch_execute(...)` signature and docstring). Immediately after argument parsing/validation, before any Odoo client call, insert:

```python
    # Read-only kill-switch — reject the whole batch if any operation is a
    # side-effect call. Runs before classification.
    from .safety import is_side_effect_method
    from .safety_profile import get_profile

    profile = get_profile()
    if profile.read_only:
        for op in operations:
            method = op.get("method", "")
            if is_side_effect_method(method):
                return BatchExecuteResponse(
                    success=False,
                    error=(
                        f"read-only mode is active: batch contains side-effect "
                        f"operation '{method}' on '{op.get('model', '?')}' "
                        f"(set MCP_READ_ONLY=false to enable writes)."
                    ),
                    results=[],
                )
```

(Place this right after the JSON parsing of `operations_json` into `operations` and before any classification or execution.)

- [ ] **Step 4: Add the guard to `execute_workflow`**

The actual `execute_workflow` signature in `src/odoo_mcp/server.py:871` is `async def execute_workflow(workflow, params_json, confirmed, confirmation_token, progress)`. All current pre-built workflows (`quote_to_cash`, `lead_to_won`, `create_and_post_invoice`) involve side-effect steps, so under read-only we refuse the whole call rather than walking individual steps.

Find the line `params = json.loads(params_json) if params_json else {}` (around line 894) and the surrounding try/except. Immediately after the try/except block that parses `params_json` (around line 901, before the `# --- Safety Classification for workflow ---` comment), insert:

```python
    # Read-only kill-switch — workflows are by definition multi-step actions.
    from .safety import is_side_effect_method  # noqa: F401  (kept for parity)
    from .safety_profile import get_profile

    profile = get_profile()
    if profile.read_only:
        return ExecuteWorkflowResponse(
            workflow=workflow,
            success=False,
            error=(
                f"read-only mode is active: workflow '{workflow}' is a "
                f"multi-step action and is rejected under MCP_READ_ONLY=true "
                f"(set MCP_READ_ONLY=false or MCP_SAFETY_MODE=strict to enable workflows)."
            ),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_read_only_guard.py -v`
Expected: PASS, all 8 cases.

- [ ] **Step 6: Commit**

```bash
git add src/odoo_mcp/server.py tests/test_read_only_guard.py
git commit -m "feat(safety): Extend MCP_READ_ONLY guard to batch_execute and execute_workflow

Both surfaces reject the entire request when any operation/step is a
side-effect call under read-only mode."
```

---

## Task 5: Write allowlist enforcement

**Files:**
- Modify: `src/odoo_mcp/safety.py` — extend `classify_operation` signature and logic
- Create: `tests/test_write_allowlist.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_write_allowlist.py`:

```python
"""Tests for MCP_WRITE_ALLOWLIST enforcement in classify_operation."""
import pytest

from odoo_mcp.safety import RiskLevel, classify_operation
from odoo_mcp.safety_profile import resolve


def _profile(env: dict):
    return resolve(env)


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
    # Method is now permitted — risk level stays HIGH, but not BLOCKED.
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
    # write is NOT in the allowlist — under enforcement, it's blocked.
    result = classify_operation("res.partner", "write", args=[[1], {"name": "X"}])
    assert result.risk_level is RiskLevel.BLOCKED


def test_no_allowlist_no_enforcement(monkeypatch):
    """Without MCP_WRITE_ALLOWLIST set and not under locked, classifier
    behaviour is unchanged (current strict-mode semantics)."""
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
    assert "security-critical" in (result.blocked_reason or "").lower() or \
           "blocked" in (result.reason or "").lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_write_allowlist.py -v`
Expected: FAIL — `classify_operation` does not yet consult the allowlist.

- [ ] **Step 3: Wire the allowlist into `classify_operation`**

Open `src/odoo_mcp/safety.py`. The `classify_operation` function currently does not import the profile. Modify it to consult the resolved profile AFTER the SAFE-method early return and BEFORE the BLOCKED_MODELS check, and keep BLOCKED_MODELS authoritative for security-critical models.

Add the helper function after `is_side_effect_method()`:

```python
def _allowlist_blocks(model: str, method: str, profile) -> bool:
    """Return True if the resolved profile's allowlist is enforced AND the
    given (model, method) is not permitted. Does NOT short-circuit BLOCKED_MODELS
    or SAFE methods — callers must check those first.
    """
    if not profile.write_allowlist_enforced:
        return False
    if not is_side_effect_method(method):
        return False
    full_key = f"{model}.{method}"
    wildcard_key = f"{model}.*"
    return (
        full_key not in profile.write_allowlist
        and wildcard_key not in profile.write_allowlist
    )
```

Modify `classify_operation` to take the profile into account. Find the existing function (around line 191) and update it:

```python
def classify_operation(
    model: str,
    method: str,
    args: list | None = None,
    kwargs: dict | None = None,
) -> SafetyClassification:
    """
    Classify an Odoo operation by risk level.

    Classification logic:
    1. SAFE_METHODS → SAFE (even on blocked/sensitive models)
    2. BLOCKED_MODELS + non-safe method → BLOCKED
    3. Allowlist enforced AND method not permitted → BLOCKED
    4. HIGH_METHODS → HIGH (always confirm)
    5. MEDIUM_METHODS → depends on mode/model/volume
    6. Unknown methods → MEDIUM
    """
    from .safety_profile import get_profile

    args = args or []
    kwargs = kwargs or {}
    mode = _get_safety_mode()
    profile = get_profile()
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

    # 2. Blocked models refuse all non-safe methods (allowlist cannot override)
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
            reason=(
                f"'{model}.{method}' is not in MCP_WRITE_ALLOWLIST."
            ),
            blocked_reason=(
                f"Side-effect call '{model}.{method}' rejected: not present in "
                f"MCP_WRITE_ALLOWLIST. Add the entry to allow it, or use a "
                f"safe read method instead."
            ),
        )

    # 4-6. (Existing logic unchanged — keep the rest of the function as-is.)

    # 4. High-risk methods always require confirmation
    if method in HIGH_METHODS:
        # ... (existing code unchanged)
```

(Keep the rest of the function body identical. Only the early section above changes.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_write_allowlist.py tests/test_safety.py -v`
Expected: PASS, allowlist tests + existing safety tests both green.

- [ ] **Step 5: Commit**

```bash
git add src/odoo_mcp/safety.py tests/test_write_allowlist.py
git commit -m "feat(safety): Enforce MCP_WRITE_ALLOWLIST in classifier

When the resolved profile enforces the allowlist (locked mode, or
explicit MCP_WRITE_ALLOWLIST under any mode), side-effect calls are
escalated to BLOCKED unless the exact 'model.method' or 'model.*'
entry is present. BLOCKED_MODELS cannot be overridden by the allowlist."
```

---

## Task 6: HTTP bind default flip in `__main__.py`

**Files:**
- Modify: `src/odoo_mcp/__main__.py:347` (line `host = os.environ.get("MCP_HOST", "0.0.0.0")`)
- Create: `tests/test_main_bind_default.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_main_bind_default.py`:

```python
"""Tests for MCP_HOST default-resolution flip under locked mode."""
import os

from odoo_mcp.safety_profile import resolve


def test_strict_mode_default_host_is_0_0_0_0():
    profile = resolve({"MCP_SAFETY_MODE": "strict"})
    assert profile.host == "0.0.0.0"


def test_locked_mode_default_host_is_localhost():
    profile = resolve({"MCP_SAFETY_MODE": "locked"})
    assert profile.host == "127.0.0.1"


def test_explicit_host_overrides_locked_default():
    profile = resolve({"MCP_SAFETY_MODE": "locked", "MCP_HOST": "0.0.0.0"})
    assert profile.host == "0.0.0.0"


def test_main_uses_profile_host(monkeypatch):
    """The HTTP bootstrap path must read host from the profile, not
    directly from MCP_HOST. We verify the integration by importing
    __main__ and checking it consults safety_profile."""
    import importlib

    from odoo_mcp import __main__ as main_mod
    importlib.reload(main_mod)

    # Smoke check: the module imports without error and exposes main.
    assert hasattr(main_mod, "main")
```

- [ ] **Step 2: Run test to verify the host-default tests pass and the integration test fails**

Run: `pytest tests/test_main_bind_default.py -v`
Expected: First three PASS (already covered by `safety_profile`), fourth PASS as smoke test (no integration check yet — adapt below).

- [ ] **Step 3: Update `__main__.py` to read host from the profile**

Open `src/odoo_mcp/__main__.py` and find the `main()` function. Change the host/port resolution at lines 347-348:

Before:
```python
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8080"))
```

After:
```python
    from .safety_profile import get_profile

    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    profile = get_profile()
    host = profile.host
    port = int(os.environ.get("MCP_PORT", "8080"))
```

The `profile.host` already encodes "explicit `MCP_HOST` if set, otherwise the mode-default", so the existing `MCP_HOST` env var continues to work; setting `MCP_SAFETY_MODE=locked` without `MCP_HOST` flips the default to `127.0.0.1`.

Also update the banner-printing path (line 360-365) so the banner displays the resolved host:

Before:
```python
        timer = threading.Timer(
            0.4, _print_startup_banner, args=(transport, host, port)
        )
```

After (no change needed since `host` is now `profile.host`).

- [ ] **Step 4: Run tests + smoke import**

Run: `pytest tests/test_main_bind_default.py -v`
Expected: PASS.

Run: `MCP_SAFETY_MODE=locked python -c "from odoo_mcp.safety_profile import get_profile; print(get_profile().host)"`
Expected output: `127.0.0.1`

Run: `MCP_SAFETY_MODE=strict python -c "from odoo_mcp.safety_profile import get_profile; print(get_profile().host)"`
Expected output: `0.0.0.0`

- [ ] **Step 5: Commit**

```bash
git add src/odoo_mcp/__main__.py tests/test_main_bind_default.py
git commit -m "feat(http): Flip MCP_HOST default to 127.0.0.1 under locked mode

main() now reads host from the resolved profile, which encodes
'MCP_HOST if set, else mode-default'. Existing deploys with
MCP_SAFETY_MODE=strict (the current default) keep binding to 0.0.0.0;
MCP_SAFETY_MODE=locked binds to 127.0.0.1 unless MCP_HOST is set
explicitly."
```

---

## Task 7: `odoo://server-status` resource

**Files:**
- Modify: `src/odoo_mcp/resources.py` (append new handler)
- Modify: `src/odoo_mcp/server.py` — add to `_RESOURCE_ROUTES`
- Create: `tests/test_server_status_resource.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_server_status_resource.py`:

```python
"""Tests for odoo://server-status resource."""
import json

import pytest


def _read_status() -> dict:
    """Read the server-status resource via the read_resource bridge."""
    from unittest.mock import MagicMock
    from odoo_mcp.server import read_resource

    # read_resource takes (uri: str, max_chars: int) per server.py signature
    raw = read_resource("odoo://server-status")
    return json.loads(raw)


def test_status_includes_version(monkeypatch):
    monkeypatch.delenv("MCP_SAFETY_MODE", raising=False)
    status = _read_status()
    assert "version" in status
    assert isinstance(status["version"], str)


def test_status_reflects_mode(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "locked")
    status = _read_status()
    assert status["safety_mode"] == "locked"
    assert status["read_only"] is True
    assert status["host"] == "127.0.0.1"
    assert status["validate_payloads"] is True


def test_status_under_strict(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "strict")
    monkeypatch.delenv("MCP_HOST", raising=False)
    monkeypatch.delenv("MCP_READ_ONLY", raising=False)
    monkeypatch.delenv("MCP_WRITE_ALLOWLIST", raising=False)
    monkeypatch.delenv("MCP_VALIDATE_PAYLOADS", raising=False)

    status = _read_status()
    assert status["safety_mode"] == "strict"
    assert status["read_only"] is False
    assert status["validate_payloads"] is False
    assert status["host"] == "0.0.0.0"
    assert status["write_allowlist"] == []


def test_status_does_not_leak_secrets(monkeypatch):
    monkeypatch.setenv("MCP_API_KEY", "supersecret-token-12345")
    monkeypatch.setenv("ODOO_API_KEY", "odoo-key-xyz")
    monkeypatch.setenv("ODOO_PASSWORD", "hunter2")

    status = _read_status()
    raw = json.dumps(status)

    assert "supersecret" not in raw
    assert "odoo-key-xyz" not in raw
    assert "hunter2" not in raw


def test_status_includes_warnings_array(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "locked")
    monkeypatch.setenv("MCP_HOST", "0.0.0.0")  # foot-gun combination
    status = _read_status()
    assert isinstance(status["warnings"], list)
    assert len(status["warnings"]) >= 1
    assert any("0.0.0.0" in w for w in status["warnings"])


def test_status_includes_posture_open(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "permissive")
    monkeypatch.setenv("MCP_HOST", "0.0.0.0")
    monkeypatch.delenv("MCP_WRITE_ALLOWLIST", raising=False)
    monkeypatch.delenv("MCP_READ_ONLY", raising=False)

    status = _read_status()
    assert status["posture_open"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server_status_resource.py -v`
Expected: FAIL — resource handler does not exist; `read_resource("odoo://server-status")` returns an unknown-resource error.

- [ ] **Step 3: Add the resource handler in `resources.py`**

Append at the end of `src/odoo_mcp/resources.py`:

```python
# ----- Server-status resource -----

@mcp.resource("odoo://server-status")
def server_status() -> str:
    """Runtime posture introspection — non-secret config visible to clients.

    Returns JSON with the resolved safety profile, transport, capabilities,
    and any foot-gun warnings. Used by monitoring scripts and by agents that
    want to know which gates are active.
    """
    return _server_status_payload()


def _server_status_payload() -> str:
    import json
    import os

    from .safety_profile import get_profile

    try:
        from importlib.metadata import version as _pkg_version
        version = _pkg_version("odoo-mcp-19")
    except Exception:
        version = "unknown"

    profile = get_profile()
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    port_raw = os.environ.get("MCP_PORT", "8080")
    try:
        port = int(port_raw)
    except ValueError:
        port = 8080
    audit = os.environ.get("MCP_SAFETY_AUDIT", "").lower() in (
        "true", "1", "yes", "on",
    )
    default_ctx_raw = os.environ.get("MCP_DEFAULT_CONTEXT", "")
    try:
        default_ctx = json.loads(default_ctx_raw) if default_ctx_raw else None
    except json.JSONDecodeError:
        default_ctx = None

    payload = {
        "version": version,
        "transport": transport,
        "host": profile.host,
        "port": port,
        "safety_mode": profile.safety_mode.value,
        "read_only": profile.read_only,
        "write_allowlist": sorted(profile.write_allowlist),
        "write_allowlist_enforced": profile.write_allowlist_enforced,
        "validate_payloads": profile.validate_payloads,
        "audit_logging": audit,
        "default_context": default_ctx,
        "posture_open": profile.posture_open,
        "warnings": list(profile.warnings),
    }
    return json.dumps(payload, indent=2)
```

- [ ] **Step 4: Add the route to `_RESOURCE_ROUTES` in `server.py`**

Open `src/odoo_mcp/server.py` and find the `_RESOURCE_ROUTES` table (search for it — it lives near the `read_resource` tool). Add an entry:

```python
    "odoo://server-status": lambda: _resources._server_status_payload(),
```

(Use the underscore-prefixed helper so the route bypasses the FastMCP resource decorator wrapping. The `_resources` import at the top of `server.py` already provides this namespace.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_server_status_resource.py -v`
Expected: PASS, all 6 cases.

- [ ] **Step 6: Commit**

```bash
git add src/odoo_mcp/resources.py src/odoo_mcp/server.py tests/test_server_status_resource.py
git commit -m "feat(resources): Add odoo://server-status posture resource

Non-secret runtime introspection: version, transport, resolved safety
profile (mode, read_only, allowlist, host, validate_payloads),
posture_open flag, and foot-gun warnings. Used by monitoring scripts
and agents to know which gates are active. Read via the existing
read_resource tool — no new tool surface."
```

---

## Task 8: Banner posture line

**Files:**
- Modify: `src/odoo_mcp/__main__.py:319-323` (the `-- Safety layer --` block)

- [ ] **Step 1: Write a smoke test**

Append to `tests/test_main_bind_default.py`:

```python
def test_banner_includes_posture_line(capfd, monkeypatch):
    """Banner should include a posture line that names the safety mode and
    read-only/host state. Smoke check via stderr capture."""
    import importlib
    from odoo_mcp import __main__ as main_mod

    monkeypatch.setenv("MCP_SAFETY_MODE", "locked")
    monkeypatch.setenv("MCP_VERBOSE", "true")

    importlib.reload(main_mod)
    main_mod._print_startup_banner("stdio", "127.0.0.1", 8080)

    err = capfd.readouterr().err
    assert "locked" in err.lower()
    assert "read-only" in err.lower() or "read_only" in err.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main_bind_default.py::test_banner_includes_posture_line -v`
Expected: FAIL — current banner says only `Mode : strict`, no posture summary.

- [ ] **Step 3: Update the banner**

Open `src/odoo_mcp/__main__.py`. Find the `-- Safety layer --` block in `_print_startup_banner` (around lines 320-323):

Before:
```python
    parts += [
        "",
        "  -- Safety layer --",
        f"  Mode          : {safety_mode}",
        f"  Audit log     : {safety_audit}",
        ...
```

After:
```python
    from .safety_profile import get_profile
    profile = get_profile()
    posture_line = (
        f"[SAFETY {profile.safety_mode.value}"
        f"{' · READ-ONLY' if profile.read_only else ' · WRITES ON'}"
        f" · BIND {profile.host}"
    )
    if profile.write_allowlist_enforced:
        posture_line += f" · ALLOWLIST {len(profile.write_allowlist)} entries"
    posture_line += "]"

    parts += [
        "",
        "  -- Safety layer --",
        f"  {posture_line}",
        f"  Mode          : {profile.safety_mode.value}",
        f"  Read-only     : {profile.read_only}",
        f"  Allowlist     : {len(profile.write_allowlist)} entries"
        + (" (enforced)" if profile.write_allowlist_enforced else ""),
        f"  Validate pl.  : {profile.validate_payloads}",
        f"  Audit log     : {safety_audit}",
        ...
```

(Keep the `...` lines that already exist — only the posture line and four detail lines are added.)

Also: any of `profile.warnings` should be appended as `WARNING       : <text>` lines after the `WARNING       : SSL verification is DISABLED` line if any exist:

```python
    for warn in profile.warnings:
        parts.append(f"  WARNING       : {warn}")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_main_bind_default.py -v`
Expected: PASS, banner test green.

- [ ] **Step 5: Visual smoke check**

Run: `MCP_SAFETY_MODE=locked python -m odoo_mcp 2>&1 | head -40`
Expected output includes a line like `[SAFETY locked · READ-ONLY · BIND 127.0.0.1 · ALLOWLIST 0 entries]`.

(Stop with Ctrl-C — the banner check is enough; we don't need the server to fully run.)

- [ ] **Step 6: Commit**

```bash
git add src/odoo_mcp/__main__.py tests/test_main_bind_default.py
git commit -m "feat(banner): Show resolved safety posture in startup banner

One-line summary plus per-flag detail. Helps colleagues see at a glance
whether they're running with writes off, allowlist enforced, and
localhost-only bind."
```

---

## Task 9: `_FIELDS_CACHE` for payload validation

**Files:**
- Modify: `src/odoo_mcp/utils.py` (append after the existing `_DOC_CACHE`)
- Create: `tests/test_fields_cache.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_fields_cache.py`:

```python
"""Tests for the live fields_get cache used by payload pre-flight."""
import time
from unittest.mock import MagicMock

import pytest

from odoo_mcp.utils import (
    _FIELDS_CACHE,
    _FIELDS_CACHE_LOCK,
    get_fields_for_model,
)


@pytest.fixture(autouse=True)
def clear_cache():
    with _FIELDS_CACHE_LOCK:
        _FIELDS_CACHE.clear()
    yield
    with _FIELDS_CACHE_LOCK:
        _FIELDS_CACHE.clear()


def test_first_call_hits_odoo():
    client = MagicMock()
    client.execute_kw.return_value = {
        "name": {"type": "char", "readonly": False, "required": True},
        "id": {"type": "integer", "readonly": True, "required": False},
    }

    fields = get_fields_for_model(client, "res.partner")

    assert "name" in fields
    assert client.execute_kw.call_count == 1


def test_second_call_hits_cache():
    client = MagicMock()
    client.execute_kw.return_value = {"name": {"type": "char"}}

    get_fields_for_model(client, "res.partner")
    get_fields_for_model(client, "res.partner")

    assert client.execute_kw.call_count == 1


def test_cache_expires_after_ttl(monkeypatch):
    client = MagicMock()
    client.execute_kw.return_value = {"name": {"type": "char"}}

    get_fields_for_model(client, "res.partner")

    # Fast-forward time past the TTL (60s).
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 61)

    get_fields_for_model(client, "res.partner")

    assert client.execute_kw.call_count == 2


def test_empty_response_not_cached():
    """An Odoo connection that returns {} should not poison the cache."""
    client = MagicMock()
    client.execute_kw.return_value = {}

    fields = get_fields_for_model(client, "res.partner")
    assert fields == {}

    # Second call must re-query (we don't trust an empty response).
    client.execute_kw.return_value = {"name": {"type": "char"}}
    fields = get_fields_for_model(client, "res.partner")
    assert "name" in fields
    assert client.execute_kw.call_count == 2


def test_lru_eviction():
    """Cache caps at 100 entries; oldest evicted first."""
    client = MagicMock()
    client.execute_kw.return_value = {"x": {"type": "char"}}

    for i in range(110):
        get_fields_for_model(client, f"model.test_{i}")

    with _FIELDS_CACHE_LOCK:
        assert len(_FIELDS_CACHE) <= 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fields_cache.py -v`
Expected: FAIL with `ImportError: cannot import name '_FIELDS_CACHE' from 'odoo_mcp.utils'`.

- [ ] **Step 3: Add the cache to `utils.py`**

Append to `src/odoo_mcp/utils.py` (after the existing `_DOC_CACHE` block):

```python
# ----- Live fields_get cache (used by payload pre-flight) -----

_FIELDS_CACHE: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_FIELDS_CACHE_LOCK = threading.Lock()
_FIELDS_CACHE_TTL = 60  # seconds — shorter than _DOC_CACHE since model
                       # schemas can change with module updates
_FIELDS_CACHE_MAX = 100


def get_fields_for_model(client, model: str) -> dict:
    """Return the fields_get response for a model, with TTL+LRU caching.

    Empty responses are NOT cached — they typically indicate a silently-failed
    Odoo connection, and we don't want to grant a write token based on
    'no fields exist therefore validation passes'.
    """
    now = time.time()
    with _FIELDS_CACHE_LOCK:
        cached = _FIELDS_CACHE.get(model)
        if cached and (now - cached[0]) < _FIELDS_CACHE_TTL:
            _FIELDS_CACHE.move_to_end(model)
            return cached[1]

    # Cache miss or expired — fetch fresh.
    fields = client.execute_kw(model, "fields_get", [], {})

    if not fields:
        # Don't cache an empty response — we don't want to remember a failure.
        return {}

    with _FIELDS_CACHE_LOCK:
        _FIELDS_CACHE[model] = (now, fields)
        _FIELDS_CACHE.move_to_end(model)
        while len(_FIELDS_CACHE) > _FIELDS_CACHE_MAX:
            _FIELDS_CACHE.popitem(last=False)
    return fields
```

(If `OrderedDict`, `threading`, `time` are not already imported at the top of `utils.py`, add them. Most likely they are, since `_DOC_CACHE` uses them.)

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fields_cache.py -v`
Expected: PASS, all 5 cases.

- [ ] **Step 5: Commit**

```bash
git add src/odoo_mcp/utils.py tests/test_fields_cache.py
git commit -m "feat(utils): Add fields_get LRU cache for payload pre-flight

60s TTL, 100 entries, locked. Empty responses are not cached so a
silently-failed Odoo connection cannot grant a token. Used in the
next commit by the payload pre-flight validator."
```

---

## Task 10: Payload pre-flight validation

**Files:**
- Modify: `src/odoo_mcp/safety.py` — add `validate_payload_against_schema()`
- Modify: `src/odoo_mcp/server.py` — call validator before issuing token in `execute_method`
- Create: `tests/test_payload_validation.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_payload_validation.py`:

```python
"""Tests for live fields_get pre-flight validation."""
from unittest.mock import MagicMock

import pytest

from odoo_mcp.safety import (
    PayloadValidationResult,
    validate_payload_against_schema,
)


def _client_with_fields(fields: dict):
    client = MagicMock()
    client.execute_kw.return_value = fields
    return client


def test_valid_payload_passes():
    client = _client_with_fields({
        "name": {"type": "char", "readonly": False, "required": True},
        "email": {"type": "char", "readonly": False, "required": False},
    })

    result = validate_payload_against_schema(
        client, "res.partner", "write", args=[[1], {"name": "X", "email": "a@b"}],
    )

    assert result.ok is True
    assert result.errors == []


def test_unknown_field_fails():
    client = _client_with_fields({"name": {"type": "char"}})

    result = validate_payload_against_schema(
        client, "res.partner", "write", args=[[1], {"name": "X", "made_up_field": 42}],
    )

    assert result.ok is False
    assert any("made_up_field" in e for e in result.errors)


def test_readonly_field_fails():
    client = _client_with_fields({
        "name": {"type": "char", "readonly": False},
        "id": {"type": "integer", "readonly": True},
    })

    result = validate_payload_against_schema(
        client, "res.partner", "write", args=[[1], {"id": 42, "name": "X"}],
    )

    assert result.ok is False
    assert any("id" in e and "readonly" in e.lower() for e in result.errors)


def test_empty_fields_response_fails():
    """A {} response indicates a connection failure or missing model — must
    NOT pass validation."""
    client = _client_with_fields({})

    result = validate_payload_against_schema(
        client, "res.partner", "create", args=[{"name": "X"}],
    )

    assert result.ok is False
    assert any("schema" in e.lower() or "fields" in e.lower() for e in result.errors)


def test_create_validates_first_arg_dict():
    client = _client_with_fields({"name": {"type": "char"}})

    result = validate_payload_against_schema(
        client, "res.partner", "create", args=[{"name": "X"}],
    )
    assert result.ok is True


def test_non_dict_payload_skipped():
    """For methods like action_confirm with no vals dict, validation is a no-op."""
    client = _client_with_fields({"name": {"type": "char"}})

    result = validate_payload_against_schema(
        client, "sale.order", "action_confirm", args=[[1]],
    )
    assert result.ok is True


def test_validator_uses_fields_cache():
    """Two calls in quick succession should share one fields_get round-trip."""
    client = _client_with_fields({"name": {"type": "char"}})

    validate_payload_against_schema(
        client, "res.partner", "write", args=[[1], {"name": "A"}],
    )
    validate_payload_against_schema(
        client, "res.partner", "write", args=[[1], {"name": "B"}],
    )
    assert client.execute_kw.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_payload_validation.py -v`
Expected: FAIL with `ImportError: cannot import name 'PayloadValidationResult' from 'odoo_mcp.safety'`.

- [ ] **Step 3: Add the validator to `safety.py`**

Append at end of `src/odoo_mcp/safety.py`:

```python
# ----- Payload Pre-flight Validation (Phase 2) -----

@dataclass(frozen=True)
class PayloadValidationResult:
    ok: bool
    errors: list[str]


def _extract_vals_dict(method: str, args: list) -> dict | None:
    """Pull the vals dict out of an operation's args, depending on method."""
    if not args:
        return None
    if method == "create":
        # create([{...}]) or create({...})
        first = args[0]
        if isinstance(first, dict):
            return first
        if isinstance(first, list) and first and isinstance(first[0], dict):
            return first[0]  # validate first record only
        return None
    if method in ("write", "copy"):
        # write([ids], {...})
        if len(args) >= 2 and isinstance(args[1], dict):
            return args[1]
    return None  # action_*, button_*, unlink: no vals dict


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
    """
    from .utils import get_fields_for_model

    args = args or []
    vals = _extract_vals_dict(method, args)
    if vals is None:
        return PayloadValidationResult(ok=True, errors=[])

    fields = get_fields_for_model(client, model)
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
            errors.append(
                f"Field '{field_name}' is readonly on '{model}' and cannot "
                f"be written."
            )

    return PayloadValidationResult(ok=not errors, errors=errors)
```

(Add `from dataclasses import dataclass` at top of `safety.py` if not already present — check imports first.)

- [ ] **Step 4: Wire validation into `execute_method`**

Open `src/odoo_mcp/server.py`. The token issuance lives at line 388-390:

```python
            payload = _payload_digest({"args": args, "kwargs": kwargs})
            if not confirmed:
                token = _issue_confirmation_token(model, method, payload)
```

Insert the pre-flight check **immediately before** the `payload = _payload_digest(...)` line, inside the `if classification.requires_confirmation:` block (line 384). The check fires only when a token would otherwise be issued, so SAFE methods skip it entirely.

```python
        if classification.requires_confirmation:
            # Phase 2: payload pre-flight against live fields_get.
            # Only when the profile asks for it AND this is a write-shaped call.
            from .safety_profile import get_profile as _get_profile  # noqa: E501
            from .safety import (
                is_side_effect_method as _is_side_effect,
                validate_payload_against_schema as _validate_payload,
            )
            _profile = _get_profile()
            if _profile.validate_payloads and _is_side_effect(method):
                _validation = _validate_payload(
                    odoo, model, method, args=args, kwargs=kwargs,
                )
                if not _validation.ok:
                    elapsed_ms = (time.time() - start_time) * 1000
                    return ExecuteMethodResponse(
                        success=False,
                        error="Payload validation failed:\n  - " + "\n  - ".join(_validation.errors),
                        hint=f"Read odoo://model/{model}/quick-schema for the field list.",
                        execution_time_ms=round(elapsed_ms, 2),
                    )

            # Bind the token to the exact (model, method, args, kwargs) seen here.
            # Args/kwargs are post-resolve_json and post-context-merge, so the digest
            # captures what would actually be sent to Odoo.
            payload = _payload_digest({"args": args, "kwargs": kwargs})
            ...  # rest of existing block unchanged
```

Local-import the helpers with leading-underscore aliases to avoid clobbering any module-scope name. The existing `_issue_confirmation_token` block downstream stays as-is.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_payload_validation.py tests/test_safety.py tests/test_token_gate.py -v`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/odoo_mcp/safety.py src/odoo_mcp/server.py tests/test_payload_validation.py
git commit -m "feat(safety): Add live fields_get payload pre-flight (Phase 2)

When MCP_VALIDATE_PAYLOADS is enabled (locked mode default), every
write/create payload is validated against the model's live fields_get
before a confirmation token is issued. Catches hallucinated fields and
readonly writes before they reach Odoo. Empty fields_get responses
(silent connection failure) refuse to issue a token."
```

---

## Task 11: Documentation update

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Configuration table**

Open `CLAUDE.md`. Find the Configuration table (around line 112-128). Add four new rows after `MCP_BOOTSTRAP_MODELS`:

```markdown
| `MCP_READ_ONLY` | No | derived from mode | `true` to globally reject all side-effect methods (writes, `action_*`, `button_*`). |
| `MCP_WRITE_ALLOWLIST` | No | empty | Comma-separated `model.method` (or `model.*`) entries permitted as side effects. Enforced under `locked`, or explicitly via this var. |
| `MCP_VALIDATE_PAYLOADS` | No | derived from mode | `true` to validate write payloads against live `fields_get` before issuing a confirmation token. |
| `MCP_HOST` (updated) | No | `0.0.0.0` (`127.0.0.1` under `locked`) | HTTP bind. Default flips to localhost when `MCP_SAFETY_MODE=locked`. |
```

Update the existing `MCP_SAFETY_MODE` row:

```markdown
| `MCP_SAFETY_MODE` | No | `strict` | `permissive`, `strict`, or **`locked`** (new). `locked` activates `MCP_READ_ONLY=true`, `MCP_WRITE_ALLOWLIST` enforcement, `MCP_HOST=127.0.0.1` default, and `MCP_VALIDATE_PAYLOADS=true`. |
```

- [ ] **Step 2: Add a new Safety section subheading**

Find the existing `## Safety layer (v1.10.0 + v1.14.0 token gate)` section. Add a new subsection after the existing token-gate description, before `Audit log via ...`:

```markdown
### Locked mode (v1.15.0)

`MCP_SAFETY_MODE=locked` adds four overlapping protections on top of the existing classifier and token gate:

| Protection | Default under `locked` | Override |
|---|---|---|
| Global write kill-switch | on | `MCP_READ_ONLY=false` |
| Side-effect allowlist enforced | on (empty list = no writes) | populate `MCP_WRITE_ALLOWLIST` |
| HTTP bind | `127.0.0.1` | `MCP_HOST=0.0.0.0` |
| Live `fields_get` payload pre-flight | on | `MCP_VALIDATE_PAYLOADS=false` |

Each protection is independently overridable. Read `odoo://server-status` to see the resolved profile at runtime, including any foot-gun warnings.

**Allowlist syntax**: `MCP_WRITE_ALLOWLIST="sale.order.action_confirm,res.partner.message_post,product.product.*"`. The wildcard matches any method on the named model. There is no `*.method` form (too easy to over-grant).
```

- [ ] **Step 3: Add `odoo://server-status` to the resources table**

Find the `## Discovery resources` table. Add a new row:

```markdown
| `odoo://server-status` | Resolved safety profile, transport, host, allowlist, warnings. Non-secret. Used by monitoring scripts and agents to know which gates are active. |
```

- [ ] **Step 4: Verify docs render**

Run: `grep -c "MCP_SAFETY_MODE=locked" CLAUDE.md`
Expected: `≥ 2` (table row + Safety section).

Run: `grep -c "odoo://server-status" CLAUDE.md`
Expected: `≥ 1`.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: Document MCP_SAFETY_MODE=locked and the new posture resource

Configuration table gains MCP_READ_ONLY, MCP_WRITE_ALLOWLIST, and
MCP_VALIDATE_PAYLOADS rows; MCP_SAFETY_MODE row mentions the new
'locked' value; Safety layer section documents what locked mode
activates and how to override each piece. odoo://server-status
appears in the discovery resources table."
```

---

## Task 12: End-to-end smoke test

**Files:**
- Create: `tests/live/test_locked_mode_live.py`

This is a script-style live test, not pytest. Mirrors `tests/live/test_safety_live.py`.

- [ ] **Step 1: Write the live test**

Create `tests/live/test_locked_mode_live.py`:

```python
"""Live integration test for MCP_SAFETY_MODE=locked.

Mutates env state. Run directly with python:
    python tests/live/test_locked_mode_live.py

Requires .env in cwd with valid Odoo credentials.
"""
import json
import os
import sys
from pathlib import Path

# Make src/ importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def _reset_env():
    for key in (
        "MCP_SAFETY_MODE", "MCP_READ_ONLY", "MCP_WRITE_ALLOWLIST",
        "MCP_HOST", "MCP_VALIDATE_PAYLOADS",
    ):
        os.environ.pop(key, None)


def test_locked_blocks_write():
    _reset_env()
    os.environ["MCP_SAFETY_MODE"] = "locked"
    from odoo_mcp.server import execute_method
    from unittest.mock import MagicMock

    response = execute_method(
        ctx=MagicMock(),
        model="res.partner",
        method="write",
        args_json='[[1], {"name": "live-test"}]',
    )
    assert not response.success
    assert "read-only" in (response.error or "").lower()
    print("✓ locked blocks write")


def test_locked_allows_safe_read():
    _reset_env()
    os.environ["MCP_SAFETY_MODE"] = "locked"
    from odoo_mcp.server import execute_method
    from unittest.mock import MagicMock

    response = execute_method(
        ctx=MagicMock(),
        model="res.partner",
        method="search_read",
        args_json='[[]]',
        kwargs_json='{"fields": ["name"], "limit": 1}',
    )
    assert response.success, f"locked should permit reads, got: {response.error}"
    print("✓ locked allows safe read")


def test_allowlist_unblocks_named_method():
    _reset_env()
    os.environ["MCP_SAFETY_MODE"] = "locked"
    os.environ["MCP_READ_ONLY"] = "false"  # Allow writes generally...
    os.environ["MCP_WRITE_ALLOWLIST"] = "res.partner.message_post"
    from odoo_mcp.safety import classify_operation, RiskLevel

    # ...but only message_post on res.partner is allowed.
    permitted = classify_operation(
        "res.partner", "message_post", args=[[1], "live test"],
    )
    blocked = classify_operation(
        "res.partner", "write", args=[[1], {"name": "X"}],
    )
    assert permitted.risk_level is not RiskLevel.BLOCKED
    assert blocked.risk_level is RiskLevel.BLOCKED
    print("✓ allowlist unblocks named method, blocks others")


def test_server_status_resource():
    _reset_env()
    os.environ["MCP_SAFETY_MODE"] = "locked"
    from odoo_mcp.server import read_resource

    raw = read_resource("odoo://server-status")
    payload = json.loads(raw)
    assert payload["safety_mode"] == "locked"
    assert payload["read_only"] is True
    assert payload["host"] == "127.0.0.1"
    print("✓ server-status reports locked posture")


def test_payload_validation_catches_bad_field():
    _reset_env()
    os.environ["MCP_SAFETY_MODE"] = "strict"  # so writes are allowed
    os.environ["MCP_VALIDATE_PAYLOADS"] = "true"
    from odoo_mcp.server import execute_method
    from unittest.mock import MagicMock

    response = execute_method(
        ctx=MagicMock(),
        model="res.partner",
        method="write",
        args_json='[[1], {"definitely_not_a_field": "x"}]',
    )
    assert not response.success
    assert "definitely_not_a_field" in (response.error or "")
    print("✓ payload pre-flight catches hallucinated field")


if __name__ == "__main__":
    test_locked_blocks_write()
    test_locked_allows_safe_read()
    test_allowlist_unblocks_named_method()
    test_server_status_resource()
    test_payload_validation_catches_bad_field()
    print("\nAll locked-mode live checks passed.")
```

- [ ] **Step 2: Run the live test**

Run: `python tests/live/test_locked_mode_live.py`
Expected: All checks print `✓` and end with "All locked-mode live checks passed."

If any check fails, the assertion error names which check; debug that one against the unit tests for the same component.

- [ ] **Step 3: Commit**

```bash
git add tests/live/test_locked_mode_live.py
git commit -m "test: Add live integration suite for locked mode

Exercises every Phase 1 + Phase 2 protection against a real Odoo:
- locked blocks writes
- locked allows safe reads
- allowlist unblocks specific method
- server-status resource reports posture
- payload pre-flight catches hallucinated field

Script-style runner like the other tests/live/ files; not collected
by pytest."
```

---

## Final verification

- [ ] **Run full unit test suite**

Run: `pytest tests/ -v --ignore=tests/live`
Expected: PASS, all unit tests green (existing + new). Coverage on new modules ≥ 90%.

- [ ] **Coverage check**

Run: `pytest tests/ --ignore=tests/live --cov=src/odoo_mcp/safety_profile --cov=src/odoo_mcp/safety --cov-report=term-missing`
Expected: `safety_profile.py` ≥ 90%, `safety.py` ≥ 80%.

- [ ] **Lint and type-check**

Run: `ruff check src/odoo_mcp tests && black --check src/odoo_mcp tests && mypy src/odoo_mcp`
Expected: clean.

- [ ] **Bump version**

Edit `pyproject.toml`:
```toml
version = "1.15.0"
```

- [ ] **Final commit**

```bash
git add pyproject.toml
git commit -m "chore: Bump version to 1.15.0"
```

- [ ] **Push branch and open PR**

Run: `git push -u origin feat/safety-locked-mode`

Then: `gh pr create --title "feat(safety): MCP_SAFETY_MODE=locked + write-path hardening" --body "$(cat <<'EOF'
## Summary
- Adds `locked` value to `MCP_SAFETY_MODE` activating four colleague-friendly defaults.
- Each protection independently overridable: `MCP_READ_ONLY`, `MCP_WRITE_ALLOWLIST`, `MCP_HOST`, `MCP_VALIDATE_PAYLOADS`.
- New `odoo://server-status` resource for runtime posture introspection.
- Backward compatible: `permissive` and `strict` semantics unchanged.

## Test plan
- [ ] Unit tests pass (`pytest tests/ --ignore=tests/live`)
- [ ] Coverage on new modules ≥ 90%
- [ ] Live test passes against a real Odoo (`python tests/live/test_locked_mode_live.py`)
- [ ] Banner shows posture line under `MCP_SAFETY_MODE=locked`
- [ ] `odoo://server-status` returns expected payload via `read_resource`
- [ ] Existing token-gate tests still pass (no regression)
EOF
)"
```

Done.
