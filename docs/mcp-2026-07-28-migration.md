# Migrating to MCP 2026-07-28 / FastMCP 4.x

**Status**: not started. The server is pinned to `fastmcp[tasks]>=3.4.6,<4` and speaks MCP `2025-11-25`.
**Written**: 2026-08-07, against FastMCP `4.0.0b2` and MCP spec `2026-07-28` (final).

This is a decision document, not a task list. It records what breaks, what it costs, and what
was already decided — so the migration can be executed later without re-doing the analysis.

---

## 0. First, the thing that is easy to get wrong

"MCP 2.0 removed FastMCP" is half true, and the half that is true does not apply here.

- The **official MCP Python SDK v2.0.0** deleted its *in-SDK* `FastMCP` class
  (`mcp.server.fastmcp.FastMCP` → `MCPServer` in `mcp.server.mcpserver`). This repo never
  used it and is unaffected by that rename.
- The **standalone `fastmcp` package** (PrefectHQ / jlowin) — our actual dependency — is
  alive and actively released: `3.4.6` stable (2026-08-05), `4.0.0b2` beta (2026-08-07).
- What genuinely changes is the **protocol underneath it**. MCP spec `2026-07-28` is final,
  and FastMCP 4.x is where it lands.

### Why there is no rush

- The spec carries a **formal 12-month minimum deprecation window** for anything it retires.
- FastMCP 4.x **negotiates protocol era per connection** — one deployment answers both the
  sessionless `2026-07-28` protocol and the older session-based handshake.
- Nothing in this server is protocol-coupled below the FastMCP layer. `safety.py` in
  particular has zero FastMCP dependency by design, which is why the blast radius is small.

### Trigger conditions — migrate when all three hold

1. FastMCP 4.0 is **stable** (not `a`/`b`; it was `4.0.0b2` when this was written).
2. The clients that matter here — Claude Desktop, Claude Code, CLORAG — actually negotiate
   `2026-07-28`. Until then, 3.4.6 serves everyone with zero loss.
3. Someone has time to run the full verification pass in §5, including the multi-user path
   against a real CLORAG registry.

Until then the `<4` pin holds and `tests/test_dependency_pins.py` enforces it.

---

## 1. Protocol delta, scoped to this server

Changes in `2026-07-28` versus `2025-11-25` that touch anything we do. (The full changelog
is at `modelcontextprotocol.io/specification/2026-07-28/changelog`; most of it is irrelevant
to us and is deliberately not reproduced here.)

| Change | Relevance |
|---|---|
| **Stateless core** — `initialize`/`notifications/initialized` handshake removed, `Mcp-Session-Id` gone; every request carries protocol version and client capabilities in `_meta` | Helps us — see §4 |
| **`server/discover`** — servers MUST implement it to advertise versions, capabilities, identity | FastMCP provides it; no work |
| **Server-initiated requests removed** — elicitation, sampling, roots replaced by Multi-Round-Trip Requests (`InputRequiredResult` / `inputRequests` / `inputResponses`) | **Breaks `configure_odoo`** — §2 |
| **Tasks moved to an extension** (`io.modelcontextprotocol/tasks`); `tasks/result` → polling via `tasks/get`, plus `tasks/update`; `tasks/list` removed | **Breaks `batch_execute` / `execute_workflow`** — §2 |
| **`CacheableResult`** — `ttlMs` + `cacheScope` required on `tools/list`, `prompts/list`, `resources/list`, `resources/read`, `resources/templates/list` | Opportunity — §4 |
| **Deterministic `tools/list` ordering** SHOULD be provided, for client and LLM prompt caching | Trivial — 5 tools |
| **`subscriptions/listen`** replaces the HTTP GET endpoint and `resources/subscribe` | We subscribe to nothing; no work |
| Resource-not-found `-32002` → `-32602` | Error-code assertions only |
| Roots, Sampling, **Logging** deprecated (12-month window) | We already log to stderr — no work |
| SSE resumability / `Last-Event-ID` removed | Not used |
| All results carry `resultType` (`"complete"` / `"input_required"`) | FastMCP handles |

---

## 2. Per-item impact

Line numbers are as of v1.15.0 and will drift — treat them as pointers, not addresses.

| # | Item | Where | Break type | Decision |
|---|---|---|---|---|
| 1 | `configure_odoo` elicitation | `server.py:714-845` | **Runtime** — `ctx.elicit()` raises on `2026-07-28`; `response_type=` becomes mandatory even on legacy connections | **Remove the tool** — §3 |
| 2 | `task=True` tools | `server.py:536`, `server.py:870` | **Startup** — needs `fastmcp_tasks.TasksExtension` registered via `mcp.add_extension(...)`, else *"Task-enabled tools require the tasks extension"* | Adopt the extension |
| 3 | Templated resources | all 27 in `resources.py` | **Behavior** — 4.x screens template params for path traversal by default | `ResourceSecurity(exempt_params=...)` |
| 4 | `mcp.types` imports | `app.py:17` (`Icon`), `skill_visibility.py:14` (`mt`) | **Import** — SDK v2 moves wire types to `mcp-types` and renames attributes to snake_case (`mimeType` → `mime_type`) behind a warn-once bridge | Update both call sites |
| 5 | `_current_transport` | `skill_visibility.py:34` | **Fragile** — private API; survives in `4.0.0b2` but unsupported, and "transport" is a weaker signal once sessionless | Replace the stdio escape hatch |
| 6 | Middleware breadth | `skill_visibility.py` | **Behavior** — 4.x middleware observes all inbound messages incl. notifications and unroutable requests; `on_list_prompts` / `on_get_prompt` still fire once per valid request | Verify only |
| 7 | `httpx` → `httpx2` | — | None — `odoo_client.py` uses `requests` | No action |
| 8 | Dependency floors | `pyproject.toml` | pydantic `>=2.12` (have 2.12.5 ✓), starlette `>=1.0.1` (have 1.4.1 ✓) | Already satisfied |

### Notes on the non-obvious ones

**#3 — why the resource screening matters.** Every `odoo://` resource takes a dotted model
name (`res.partner`), and `odoo://bundle/{m1,m2,...}` takes a comma-separated list. The 4.x
default rejects `..` segments and absolute paths. Dotted model names are not `..`, so most
should pass — but this must be **tested, not assumed**, particularly for the bundle route.
Exempting is safe here because `_validate_model` already regex-gates every param against
`^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$` (max 128) before it reaches Odoo. Rejected reads
return a non-leaky resource-not-found.

**#5 — what the stdio hatch is actually for.** `SkillVisibilityMiddleware._allowed_skills()`
returns `None` (unrestricted) when the transport is stdio, so Alan's local session sees every
`cyanview-*` prompt. Under a sessionless protocol the useful question is no longer "which
transport?" but "is there an authenticated identity?" — and the code already handles the
HTTP cases directly below (missing token → fail closed; `ENV_ADMIN_CLIENT_ID` or
`role == "admin"` → unrestricted). Prefer keying on **absence of an auth provider** rather
than transport identity. Whatever replaces it must preserve the current fail-closed
behavior, which `tests/test_skill_visibility.py` pins.

---

## 3. `configure_odoo`: remove it

**Decided 2026-08-07.** Do not port it to the MRTR guard-tool pattern.

Rationale:

- `python -m odoo_mcp --setup` already covers the job, and covers it better — it generates
  `.env`, the Docker command, and the Claude Desktop config.
- In multi-user mode credentials come from the CLORAG registry. A runtime-configure tool has
  no role there at all.
- The MRTR rewrite is the single most expensive item in this document and depends on
  client-side MRTR support that is brand new.

Removal touches: `server.py:127` (module comment), `server.py:714-845` (the tool, its
`ElicitationResult` model, and the `fastmcp.server.elicitation` import), `CLAUDE.md:10,71,170,177`,
`README.md:13,227,234`. Tool surface goes **5 → 4**.

The startup banner counts are introspected from the FastMCP instance (`__main__.py:223,330`),
so they follow automatically — but confirm at runtime rather than trusting it.

---

## 4. Opportunities, not just repairs

Two of these are worth doing on their own merits, independent of the forced changes.

### `ttlMs` / `cacheScope` — the real prize

This server is unusually well suited to result caching. It exposes **27 discovery resources**
serving near-static schema data — `quick-schema`, `fields`, `schema`, `workflow`, `bundle`,
`session-bootstrap`, plus the reference docs (`domain-syntax`, `aggregation`, `pagination`,
`hierarchical`) which are effectively immutable between releases. The server already runs a
`_DOC_CACHE` (5-min TTL, 100-entry LRU) internally; `ttlMs` extends that saving to the client
and removes the round-trip entirely rather than just the Odoo call.

Suggested tiering when this is implemented:

- **Static reference docs** (`domain-syntax`, `aggregation`, `pagination`, `hierarchical`,
  `templates`) — long `ttlMs`, `cacheScope: "public"`. They contain no tenant data.
- **Schema resources** (`quick-schema`, `fields`, `schema`, `methods`, `bundle`) — moderate
  `ttlMs`. `cacheScope: "private"`: schemas reflect the connected user's field visibility,
  and in multi-user mode two callers are genuinely different Odoo users.
- **Runtime-varying** (`model-limitations`, which merges runtime-detected issues) — short
  `ttlMs` or none.

`cacheScope: "public"` on anything user-dependent would leak one tenant's schema to another
through a shared intermediary. When in doubt, `"private"`.

Deterministic `tools/list` ordering is nearly free at 5 tools and improves LLM prompt-cache
hit rates; do it in the same pass.

### Statelessness suits the CLORAG deployment

Bearer-token-per-request already matches the sessionless model — `DbTokenVerifier` hashes and
looks up the token on every call and holds no session state. Dropping `Mcp-Session-Id` means
no session affinity, so the multi-user HTTP deployment can scale behind a plain round-robin
load balancer. Nothing to build; note it when capacity becomes a question.

### FastMCP 4.x enterprise auth (SEP-990)

FastMCP 4.x ships role-based access control that may subsume part of the hand-rolled role-claim
layer in `auth_verifier.py` / `safety.py`. **Evaluate, do not assume.** The registry schema is
a CLORAG-owned contract (`~/dev/clorag`, `core/user_db.py`) and any change starts there, not
here. The current design deliberately keeps `safety.py` free of framework dependencies; do not
trade that away for framework convenience without a concrete gain.

---

## 5. Verification for the migration

Everything in Phase 1's gate (`uv run pytest tests/ --ignore=tests/live`, ruff, mypy, both
transports, capability counts), plus:

- **Dual-era**: connect a `mode="legacy"` client and a `2026-07-28` client to the same
  running server; both must work. This is the whole premise of migrating without a flag day.
- **Tasks**: `batch_execute` and `execute_workflow` must register at startup *and* report
  progress through the new `tasks/get` polling flow.
- **Resource templates**: read all 27 URIs, with attention to `odoo://bundle/{m1,m2,...}` and
  any dotted model name, to confirm the path-traversal screen does not reject legitimate params.
- **Multi-user, against a real CLORAG registry** (not just the seeded fixture): a registry
  bearer token resolves to the caller's personal Odoo client; `readonly` still blocks every
  non-safe method; `cyanview-*` filtering still applies in **both** `list_prompts` and
  `get_prompt`; a missing token still fails closed.
- **Token gate**: the `2026-07-28` payload-digest path must still bind a confirmation token to
  `(model, method, payload_digest)`. `tests/test_token_gate.py` pins this; make sure it is not
  quietly weakened by any argument-handling change.
- Update `tests/test_dependency_pins.py` majors, and this document's status line, last.

---

## 6. Known unrelated breakage

`tests/live/test_v1110_live.py` fails at TEST 2 with
`AttributeError: module 'odoo_mcp.constants' has no attribute '_DEFAULT_CONTEXT'`.

Pre-existing and unrelated to any of the above: commit `4a39f13` ("Read safety env vars at
call time") replaced the module-level `_DEFAULT_CONTEXT` constant with the call-time
`_parse_default_context()` / `get_default_context()` pair, and this live runner was never
updated. It monkey-patches the old attribute to test `_merge_context`. Fixing it means
patching `MCP_DEFAULT_CONTEXT` in the environment instead — which is how
`tests/test_safety.py` already does it. Worth doing before relying on this runner as a
migration gate.
