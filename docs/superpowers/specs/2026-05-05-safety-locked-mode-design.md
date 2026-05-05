# Design: Safety profile `locked` and write-path hardening

- **Status**: Approved 2026-05-05, pending implementation
- **Target version**: v1.15.0
- **Scope**: Phase 1 (defaults + allowlist + bind + posture resource) and Phase 2 (live `fields_get` pre-flight)

## Goal

Make `odoo-mcp-19` safe for colleagues to use against their own Odoo instances without requiring them to read the full safety section of CLAUDE.md to avoid foot-guns. The threat model is **an over-eager AI agent acting through a colleague's MCP server**, not a hostile external attacker — `MCP_API_KEY` already covers the latter.

Today's defaults assume a careful operator. We want defaults that fail safe when the operator is distracted or new.

## Non-goals

- Multi-tenant SaaS exposure hardening — that lives in `saas-mcp-odoo`.
- Replacing the existing token-digest safety gate (v1.14.0). The new layers sit *in front of* it; they do not change how confirmation tokens are issued or validated.
- Adding a 3-step preview/validate/execute write workflow (the tuanle96 pattern). The existing 2-step gate plus payload pre-flight gives most of the same guarantees with less friction.

## Architecture: hybrid umbrella + per-flag override

One umbrella variable carries an opinion about the whole posture; individual variables override pieces of it.

- **Umbrella**: `MCP_SAFETY_MODE` (existing) gains a third value `locked`.
- **Per-flag overrides**: `MCP_READ_ONLY`, `MCP_WRITE_ALLOWLIST`, `MCP_HOST`, `MCP_VALIDATE_PAYLOADS`. Each, if set explicitly, wins over what the profile would have produced.
- **Resolution order at startup**: profile defaults → individual env-var overrides → final values used everywhere downstream.

Resolution happens once in a new `safety_profile.resolve()` step called from `__main__.py` before the FastMCP app starts. Downstream code (`safety.py`, `server.py`, `__main__.py` HTTP bootstrap) reads the resolved values, not the raw env vars. Existing call-time-env-read for `MCP_SAFETY_MODE`/`MCP_SAFETY_AUDIT`/`MCP_DEFAULT_CONTEXT` is preserved — those continue to be re-read on each call so tests can monkeypatch.

## Profile-to-defaults table

| | `permissive` | `strict` (default) | `locked` (new) |
|---|---|---|---|
| Classifier behavior | HIGH/BLOCKED confirm | + MEDIUM confirm | + MEDIUM confirm |
| `MCP_READ_ONLY` default | `false` | `false` | `true` |
| `MCP_WRITE_ALLOWLIST` enforced | no | no | yes |
| `MCP_HOST` default | `0.0.0.0` | `0.0.0.0` | `127.0.0.1` |
| `MCP_VALIDATE_PAYLOADS` default | `false` | `false` | `true` |

Backward compatibility: `permissive` and `strict` keep their current behavior exactly. No existing deploy changes posture on upgrade unless the operator opts in to `locked`.

## Components

### `MCP_SAFETY_MODE=locked`

Adds a new value to the existing variable. The new value is the umbrella — when set, it flips the four downstream defaults shown in the table. No new code path; the resolver just reads it and emits resolved values.

### Side-effect method definition

For both `MCP_READ_ONLY` and `MCP_WRITE_ALLOWLIST`, "side-effect method" means any of:

- The literal names `create`, `write`, `unlink`, `copy`, `action_archive`, `action_unarchive`.
- Any method matching `action_*`, `button_*`, `_action_*` (private action hooks blocked anyway, but listed for completeness).
- Any method appearing as a transition in `MODEL_STATE_MACHINES` (`action_confirm`, `action_post`, `button_validate`, etc.).

This list is encoded as a single predicate `is_side_effect_method(method)` in `safety_profile.py`. It does not require running the full classifier — it's a cheap pattern match. The classifier still runs for finer-grained risk levels (MEDIUM vs HIGH vs BLOCKED), but the read-only and allowlist gates use the predicate directly.

### `MCP_READ_ONLY=true|false`

When `true`, any call where `is_side_effect_method(method)` returns `true` is rejected before classification with a clear error: `read-only mode is active (set MCP_READ_ONLY=false to enable writes)`. SAFE methods (`read`, `search_read`, `name_search`, `fields_get`, `default_get`, `search_count`, etc.) are unaffected. Enforcement point: the top of `execute_method`, `batch_execute`, and `execute_workflow`, before the classifier runs.

### `MCP_WRITE_ALLOWLIST="model.method,model.method"`

Comma-separated list of `<model>.<method>` entries. When the resolved profile says the allowlist is enforced (i.e. `locked` mode, or `MCP_WRITE_ALLOWLIST` set explicitly under any mode), any side-effect method must appear in the list or the call is rejected. Empty list under enforcement = no side-effect methods allowed (equivalent to `MCP_READ_ONLY=true`).

Wildcard support: `model.*` matches any method on the model. We do *not* support `*.method` — too easy to over-grant.

Enforcement point: in `safety.classify_operation`, after the existing classifier produces a level. If the level is anything other than SAFE and the allowlist is enforced, the entry must be present, otherwise classification escalates to BLOCKED with a sanitized error message naming the missing entry.

### `MCP_HOST` default flip in `locked`

No new env var. The HTTP bootstrap in `__main__.py` already reads `MCP_HOST` with a default of `0.0.0.0`. We change that default-resolution: when `MCP_SAFETY_MODE=locked` and `MCP_HOST` is unset, default to `127.0.0.1`. Setting `MCP_HOST=0.0.0.0` explicitly under `locked` still works — the operator opts in to remote bind. The startup banner names the actual bind so colleagues can see what they got.

### `MCP_VALIDATE_PAYLOADS=true|false` (Phase 2)

When `true`, before issuing a confirmation token for any write or `create` operation, the safety layer fetches the target model's `fields_get` (cached for 60s, lock-protected like `_DOC_CACHE`) and validates that:

1. The model exists in `fields_get` and the response is non-empty (catches a silently-failed Odoo connection that returns `{}`).
2. Every key in the operation's `vals` dict is a real, non-`@api.private` field on the model.
3. Readonly fields are not being written.
4. `Many2one` references resolve to integer IDs (or have a corresponding `resolve_json` directive).

If validation fails, no token is issued. The error names the offending fields and gives a hint pointing to `odoo://model/{model}/quick-schema`. Validation runs *before* `_issue_confirmation_token`, so a hallucinated payload never receives a token.

This adds one round-trip to Odoo before each first-time write to a model, but the `fields_get` cache means subsequent writes in the same window are validated locally.

### Resource: `odoo://server-status`

A new resource handler in `resources.py` and route entry in `_RESOURCE_ROUTES`. Returns a structured payload with non-secret runtime posture:

```json
{
  "version": "1.15.0",
  "transport": "stdio",
  "host": "127.0.0.1",
  "port": 8080,
  "safety_mode": "locked",
  "read_only": true,
  "write_allowlist": ["sale.order.action_confirm", "res.partner.message_post"],
  "validate_payloads": true,
  "audit_logging": true,
  "default_context": {"lang": "fr_FR"},
  "posture_open": false,
  "warnings": []
}
```

`posture_open` is `true` when running under `permissive` mode AND `MCP_HOST=0.0.0.0` AND no allowlist — i.e. the maximally-open configuration. The `warnings` array surfaces any "you're running with a foot-gun" combinations explicitly so monitoring scripts can alert.

No secrets in the payload. `MCP_API_KEY` is never echoed. `ODOO_*` connection details are not echoed (those go in the existing connection-summary banner at startup).

Read via the existing `read_resource("odoo://server-status")` tool — no new tool surface.

### Startup banner update

The existing verbose banner gains a single posture line, e.g.:

```
[SAFETY locked · READ-ONLY · BIND 127.0.0.1 · ALLOWLIST 0 entries]
```

For `permissive`/`strict` the line is shorter:

```
[SAFETY strict · WRITES ON · BIND 0.0.0.0]
```

This is the on-the-glass equivalent of the `odoo://server-status` resource.

## Data flow

```
startup
  └─ safety_profile.resolve(env) → ResolvedProfile
        ├─ safety_mode: "locked"
        ├─ read_only: true
        ├─ write_allowlist: frozenset({"sale.order.action_confirm", ...})
        ├─ host: "127.0.0.1"
        ├─ validate_payloads: true
        └─ audit: true

per-call (execute_method)
  ├─ if profile.read_only and is_side_effect_method(method) → reject early
  ├─ classify_operation(model, method, args, kwargs, profile)
  │     └─ if profile.write_allowlist enforced and is_side_effect_method(method)
  │           and f"{model}.{method}" not in allowlist → escalate to BLOCKED
  ├─ if profile.validate_payloads and is_side_effect_method(method) → fields_get pre-flight
  │     └─ on failure: no token, sanitized error
  └─ existing token-digest gate path (unchanged)
```

## File-level changes

| File | Change | Approx LOC |
|---|---|---|
| `src/odoo_mcp/safety_profile.py` | **NEW** — `ResolvedProfile` dataclass + `resolve(env)` function | ~120 |
| `src/odoo_mcp/safety.py` | Accept resolved profile; honor `read_only`, `write_allowlist`, `validate_payloads` in `classify_operation` and the gate path | ~80 added |
| `src/odoo_mcp/server.py` | Early read-only short-circuit in `execute_method`/`batch_execute`/`execute_workflow`; payload pre-flight call | ~60 added |
| `src/odoo_mcp/resources.py` | New `odoo://server-status` handler | ~50 |
| `src/odoo_mcp/__main__.py` | Resolve profile at startup; flip `MCP_HOST` default under `locked`; banner posture line | ~40 changed |
| `src/odoo_mcp/constants.py` | New constants for side-effect method patterns used by read-only check | ~15 added |
| `src/odoo_mcp/utils.py` | `_FIELDS_CACHE` (60s TTL, locked, LRU 100) for payload pre-flight | ~50 added |
| `tests/test_safety_profile.py` | **NEW** — resolver matrix, override precedence | ~150 |
| `tests/test_read_only.py` | **NEW** — read-only kill-switch enforcement | ~80 |
| `tests/test_write_allowlist.py` | **NEW** — allowlist matching, wildcards, missing-entry errors | ~120 |
| `tests/test_payload_validation.py` | **NEW** — Phase 2 fields_get pre-flight (with mocked client) | ~150 |
| `tests/test_server_status_resource.py` | **NEW** — resource payload shape, no-secrets invariant | ~80 |
| `CLAUDE.md` | Document the new mode, vars, resource | ~30 changed |

`server.py` is already 1262 LOC and bumping it further is mild pressure on its 800-LOC soft target. The new logic is ~60 LOC of guard checks and is the right home for it (these are tool-boundary concerns), so we accept the bump and revisit splitting `execute_method` out of `server.py` only if it crosses 1500.

## Testing strategy

- **Unit (no Odoo, run in CI)**: `safety_profile.resolve` matrix covers all 3 modes × all 4 override permutations. `read_only` and `write_allowlist` enforcement tested with mocked classifier inputs. `server-status` resource tested for shape and no-secret invariant.
- **Live (manual, gated on `.env`)**: `tests/live/test_locked_mode_live.py` exercises the full path against a real Odoo: a SAFE read works, a write under `MCP_READ_ONLY=true` is rejected, an allowlisted `action_confirm` succeeds, a non-allowlisted method is rejected with the right error, payload pre-flight catches a hallucinated field.
- **Coverage target**: the new code in `safety_profile.py` and the new branches in `safety.py` must be ≥ 90% covered by unit tests. The rest follows the existing 80% target.

## Backward compatibility

- All existing env vars keep their current semantics. `MCP_SAFETY_MODE=permissive` and `MCP_SAFETY_MODE=strict` (the current default) behave exactly as today.
- Adding `locked` is opt-in. No deploy changes posture on upgrade.
- The `MCP_HOST` default flip applies *only* when `MCP_SAFETY_MODE=locked`. Existing Docker deploys (`MCP_SAFETY_MODE=strict` by default) continue to bind `0.0.0.0` and listen on the same port.
- The new `odoo://server-status` resource is additive.
- The Phase 2 `MCP_VALIDATE_PAYLOADS` defaults to `false` under `permissive`/`strict`, so existing clients see no behavior change.

## Open questions

None blocking — the user approved the naming and the `MCP_HOST` default-flip policy on 2026-05-05.

## What this design intentionally does NOT include

- **A 3-tool write ritual** (preview/validate/execute). The token-digest gate plus payload pre-flight covers the same payload-tampering and hallucinated-field threats.
- **A separate `health_check` tool**. Resources are for state, tools are for actions. `odoo://server-status` is the right surface.
- **A new "broad mode" master kill-switch**. We rely on classifier + allowlist + read-only as overlapping defenses. A single broad-mode flag is harder to reason about than the resolved profile.
- **Per-method or per-model risk tunables**. `MCP_WRITE_ALLOWLIST` already gives operators that lever; a more elaborate config schema would be over-engineering for the colleague threat model.
