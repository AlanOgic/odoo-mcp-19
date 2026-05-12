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
import re
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


_ALLOWLIST_ENTRY_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+\.([\w]+|\*)$")


def _parse_allowlist(raw: str | None) -> frozenset[str]:
    """Parse a comma-separated 'model.method' list into a frozenset.

    Each entry must match either 'model.method' or 'model.*'. Entries that
    do not match are dropped with a warning — this catches mistakes like
    '*.write' (cross-model wildcards are deliberately not supported because
    they over-grant) and obvious typos.
    """
    if not raw:
        return frozenset()
    valid: set[str] = set()
    for entry in raw.split(","):
        cleaned = entry.strip()
        if not cleaned:
            continue
        if _ALLOWLIST_ENTRY_RE.match(cleaned):
            valid.add(cleaned)
        else:
            logger.warning(
                "MCP_WRITE_ALLOWLIST entry %r is not in 'model.method' or "
                "'model.*' form — dropped (cross-model wildcards like "
                "'*.method' are not supported).",
                cleaned,
            )
    return frozenset(valid)


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
    if mode is SafetyMode.LOCKED and not read_only:
        warnings.append(
            "MCP_SAFETY_MODE=locked but MCP_READ_ONLY=false — "
            "the primary write kill-switch is disabled; only "
            "MCP_WRITE_ALLOWLIST gates side-effect calls."
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
