# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**odoo-mcp-19** — Standalone MCP server for Odoo 19+ using the **v2 JSON-2 API** (`POST /json/2/{model}/{method}`, Bearer token auth, named args only). No v1 fallback.

- **Version**: 1.16.0 · **Python**: 3.10+ · **MCP**: 2025-11-25 (FastMCP `>=3.4.6,<4`; `cryptography>=42` is a *direct* dependency of `token_crypto`, not just a transitive Authlib one)
- **The `<4` ceiling is deliberate.** FastMCP 4.x targets MCP spec 2026-07-28 and is not a drop-in — see `docs/mcp-2026-07-28-migration.md` and `tests/test_dependency_pins.py`, which fails the build if the bound is widened or the environment drifts. **`uv.lock` is gitignored** (`.gitignore`: *"project uses pip + pyproject.toml"*) — it pins only your local `.venv`, never the documented `pip install git+…` path, so `pyproject.toml` is the only thing between a fresh install and FastMCP 4.x. Do not relax the ceiling on the strength of the lockfile.
- **Surface**: 5 tools, 28 `odoo://` resources, 19 prompts (12 generic + 7 `cyanview-*` workflow skill prompts)
- **Discovery is via resources, action is via tools** — there is no `list_models` tool, agents read `odoo://models` instead.
- **Two deployment shapes**: single-user (STDIO or HTTP with one static `MCP_API_KEY`) and **multi-user HTTP** (per-user `cv_odoo_…` keys from the CLORAG-managed registry, personal Odoo clients, per-user skill visibility — see "Multi-user mode" below).

## Development commands

```bash
# Install (local workflow — uv-managed .venv; uv.lock is deliberately gitignored, the project ships via pip + pyproject.toml)
uv sync --extra dev
# then prefix commands with `uv run`, e.g.:
uv run pytest tests/test_safety.py
# pip equivalent: pip install -e ".[dev]"

# Install as a package (installs the `odoo-mcp-19` console entry point) — deliberately NOT on PyPI
pip install git+https://github.com/AlanOgic/odoo-mcp-19.git

# Run server — STDIO (default); loads .env from cwd
python -m odoo_mcp

# Run server — HTTP (requires MCP_API_KEY *or* USERS_DB_PATH, else sys.exit(1))
MCP_TRANSPORT=streamable-http MCP_API_KEY=<token> python -m odoo_mcp

# Interactive setup wizard — generates .env, Docker cmd, Claude Desktop config
python -m odoo_mcp --setup

# Tests — unit (no Odoo needed)
uv run pytest tests/ --ignore=tests/live
uv run pytest tests/test_safety.py::TestClassifyOperation::test_safe_methods_are_safe   # single test
# Note: bare `pytest tests/` collects NO live tests (they are __main__-guarded scripts with no test_* functions),
# but it does *import* them, and their module-level load_dotenv() pushes real .env credentials into os.environ
# for the whole session. Always --ignore=tests/live.

# Tests — live (requires .env with real Odoo creds; script-style runners)
python tests/live/test_safety_live.py
python tests/live/test_v1110_live.py
python tests/live/test_locked_mode_live.py

# Format + lint + typecheck
black . && isort .   # both clean; isort has skip_gitignore=true so it skips venvs like black does
ruff check .         # known baseline: 33 errors (27 E501, 6 E402) — see below
mypy src/odoo_mcp    # known baseline: 75 errors, mostly [index]/[assignment] in resources.py + server.py

# If `uv run pytest` reports ModuleNotFoundError: No module named 'odoo_mcp' (17 collection
# errors), the dev extras are not installed — run `uv sync --extra dev` first. uv also ignores a
# VIRTUAL_ENV pointing anywhere other than ./.venv (it warns and uses .venv regardless), so an
# activated sibling venv silently gives you a system pytest/ruff/mypy or none at all.

# Docker
docker build -t odoo-mcp-19 .
docker compose up -d           # uses .env, requires MCP_API_KEY
# Multi-user overlay (on the clorag host — registry mounted read-only)
docker compose -f docker-compose.yml -f docker-compose.multiuser.yml up -d
```

Note: live tests under `tests/live/` are **script-style runners**, not pytest modules — invoke them directly with `python`. They mutate environment state.

- **Unit (no Odoo)**: everything in `tests/` except `tests/live/` — run with `pytest --ignore=tests/live`. Highlights: `test_resources.py` patches `get_odoo_client` with a stub (pins resource-layer validation/error handling); `test_arg_mapping.py` pins the positional → JSON-2 named-arg contract; `test_odoo_client.py` pins bearer auth + no `result`-envelope unwrap; `test_token_gate.py` / `test_safety.py` / `test_safety_role.py` cover the gate and role-based classification; the multi-user tests (`test_auth_verifier.py`, `test_token_crypto.py`, `test_user_clients.py`, `test_skill_visibility.py`, `test_skill_prompts.py`) use the `users_db_seed` fixture in `tests/conftest.py`, which builds a temp registry with the **exact CLORAG DDL and crypto contract** — keep that fixture contract-true. Locked mode (v1.16.0) is pinned by `test_safety_profile.py` (env → profile resolution), `test_read_only_guard.py`, `test_write_allowlist.py`, `test_payload_validation.py`, `test_side_effect_predicate.py`, `test_fields_cache.py`, `test_main_bind_default.py`, and `test_server_status_resource.py`.
- **Live (need `.env`)**: anything under `tests/live/` — run with `python <file>`, not pytest.

**CI**: two workflows in `.github/workflows/`:
- `tests.yml` — on every PR and push to master: the unit suite (`pytest tests/ --ignore=tests/live`) across Python 3.10–3.13 (matrix mirrors the pyproject classifiers), plus a quality job where `black --check` and `isort --check-only` **block** while `ruff` and `mypy` run `continue-on-error` (pre-existing backlog — flip them blocking once their counts hit zero). CI deliberately provides no `ODOO_*` env and no lockfile (`uv sync` resolves fresh, mirroring what `pip install git+…` gives a user), so unit tests that quietly need a live server or drifting dependencies surface as failures.
- `claude-code-review.yml` — automated Claude code review on every PR.

Run `black . && isort .` before pushing — CI enforces formatting.

**Lint and typecheck are not clean gates.** Baseline as of v1.16.0: `pytest --ignore=tests/live` → **382 passed**; `black --check .` and `isort --check-only .` → **clean**; `ruff check .` → **33 errors** (27 E501, 6 E402); `mypy src/odoo_mcp` → **75 errors** (38 `resources.py`, 24 `server.py`, 10 `utils.py`, 1 each in `prompts.py` / `constants.py` / `user_clients.py`; counts drift slightly with the mypy version — 2.1.0 here). Judge a change by *no new errors against that baseline*, not by a zero exit code. All 6 remaining E402s are in `tests/live/`, where `load_dotenv()` must run before the `odoo_mcp` imports — inherent to those script-style runners, not a defect. None of them come from the deliberate "import `app.py` first" ordering in `server.py`/`resources.py`, so do not reorder module imports to chase them.

## High-level architecture

The package was split in v1.14.0 (commit `ea10d79`) from a 3762-line `server.py` into focused modules; the multi-user layer (commit `4d1b7f8`) added the `skill_prompts`, `auth_verifier`, `users_db`, `user_clients`, `token_crypto`, and `skill_visibility` modules (see the tree below). Import order matters: `app.py` must be imported first so the `mcp` decorator is bound before `server.py`, `resources.py`, `prompts.py`, and `skill_prompts.py` register their handlers. `app.py` also wires auth (`_get_auth_provider`) and the skill-visibility middleware at import time, based on env.

```
src/odoo_mcp/
├── __main__.py        CLI entry: STDIO/HTTP bootstrap, --setup wizard, HTTP auth preflight
├── app.py             FastMCP instance + icon + auth provider selection + middleware wiring — imported first
├── server.py          5 tools + _RESOURCE_ROUTES table + search_read fallback + safety integration
├── resources.py       28 odoo:// resource handlers
├── prompts.py         12 generic guided prompts
├── skill_prompts.py   7 cyanview-* workflow prompts, bodies loaded from skills/*.md (frontmatter stripped)
├── safety.py          Risk classification + token gate + role-based blocking + read-only/
│                      allowlist guards; `is_side_effect_method()` is the write predicate
├── safety_profile.py  `ResolvedProfile` — folds MCP_SAFETY_MODE + MCP_READ_ONLY +
│                      MCP_WRITE_ALLOWLIST + MCP_HOST + MCP_VALIDATE_PAYLOADS into one
│                      posture with foot-gun warnings; backs `odoo://server-status`
├── odoo_client.py     v2 JSON-2 client (thread-safe singleton, sanitized errors, always-Bearer auth)
│                      get_odoo_client() dispatches: per-user client if registry user, else env singleton
├── auth_verifier.py   DbTokenVerifier — sha256(token) lookup in registry; static MCP_API_KEY → env-admin identity
├── users_db.py        Read-only (mode=ro) SQLite access to the CLORAG registry; schema contract in docstring
├── user_clients.py    Per-user OdooClient cache (300s TTL, re-checks creds updated_at) + current_role()
├── token_crypto.py    Fernet/PBKDF2 decrypt of registry secrets — MUST match CLORAG token_encryption.py
├── skill_visibility.py  Middleware filtering cyanview-* prompts by the user_skills allowlist
├── arg_mapping.py     Positional → named args for 30 ORM methods + record-bound `ids` fallback;
│                      raises ValueError on any positional it has no JSON-2 name for
├── constants.py       Limits, regex validators, MODEL_STATE_MACHINES, default context
├── models.py          Pydantic response schemas (structured output)
├── utils.py           Compact schema builder, error suggestions, and two TTL+LRU caches:
│                      `_DOC_CACHE` (/doc-bearer, 300s) and `_FIELDS_CACHE` (live fields_get via
│                      `get_fields_for_model()`, 60s — shorter, it backs the locked-mode payload pre-flight)
├── skills/*.md        Packaged Cyanview skill bodies (copied from curated ~/.claude/skills/cyanview-*)
└── module_knowledge.json   Special methods for 13 modules + the static `model_limitations` block
                            (loaded at startup, shipped as package data)
```

### Cross-cutting flows

**1. Tool call → Odoo round-trip** (`execute_method`):
1. Validate model (`_validate_model`) and method (`_validate_method`) — regex + length, else 400.
2. Block `@api.private` methods statically (PRIVATE_METHOD_HINTS) and dynamically (live `/doc-bearer/`).
3. Classify via `safety.classify_operation` → SAFE / MEDIUM / HIGH / BLOCKED.
4. If gate triggers → return `pending_confirmation=true` with a single-use, 120s, op-bound `confirmation_token` in `hint`. Caller must re-call with both `confirmed=true` AND that token. **`confirmed=true` alone does not bypass the gate** — this is the v1.14.0 hardening.
5. Merge `MCP_DEFAULT_CONTEXT` into kwargs (explicit context wins).
6. Resolve Many2one names via `resolve_json` (uses `name_search`, validates target model against BLOCKED_MODELS).
7. Convert positional → named args via `arg_mapping`.
8. Send to Odoo. On 500 from `search_read` → automatically fall back to `search` + `read`, categorize the error (timeout / relational_filter / computed_field / …), record runtime issue, return enriched `issue_analysis`.
9. On any error: match against ~25 patterns in `utils.get_error_suggestion` (with `{model}` templating). Server tracebacks are logged to stderr, **never** forwarded to clients.

**2. Resource bridge** — `read_resource(uri)` exists because some clients (Claude Desktop) don't speak resource templates. The `_RESOURCE_ROUTES` table in `server.py` maps URIs to the same handlers `resources.py` registers, so the same `odoo://...` URI works either way. Two properties nothing tests: **parity** — 28 routes ↔ 28 `@mcp.resource` handlers today, and adding a resource without adding a route silently 404s the bridge while the template client keeps working; and **order** — the list is matched top-down, so `odoo://module-knowledge/{name}` must stay above `odoo://module-knowledge`, and the specific `odoo://model/{m}/…` variants above the catch-all `odoo://model/{m}`. `read_resource` also **truncates at 15 000 chars** (`_READ_RESOURCE_MAX_CHARS`), appending a `_truncated` JSON marker — so `odoo://model/{m}/schema` (~300 KB) returns a fragment unless the caller passes `max_chars=0`.

**3. Live doc enrichment** — `odoo://methods/{model}` and `@api.private` detection both consult `/doc-bearer/<model>.json` (provided by Odoo's `api_doc` module, requires `api_doc.group_allow_doc` on the API user). Cached in `_DOC_CACHE`: 5-min TTL, 100-entry LRU, `threading.Lock`. Falls back silently to static data if unavailable.

**4. Background tasks** — `batch_execute` and `execute_workflow` use FastMCP's `[tasks]` extra (backed by `pydocket` on the 3.x line) to support async execution with progress reporting (MCP 2025-11-25 SEP-1686). In FastMCP 4.x the extra becomes the separate `fastmcp-tasks` package and `task=True` tools additionally require an explicit `mcp.add_extension(TasksExtension())` — without it they fail to register at startup. This is one of the reasons for the `<4` pin.

**5. Multi-user request path** (HTTP + `USERS_DB_PATH`): bearer token → `DbTokenVerifier` hashes it (sha256) and looks it up in the registry (`server='odoo'`, non-revoked key, active user) → `AccessToken` carries `client_id=user_id` and a `role` claim → every tool call passes `role=current_role()` into safety classification, and `get_odoo_client()` resolves the caller's **personal** OdooClient (their Odoo username + decrypted API key), so writes are attributed to the real Odoo account. STDIO and the static `MCP_API_KEY` fallback bypass all of this and use the env singleton — existing single-user behavior is unchanged.

## Multi-user mode (CLORAG registry)

Activated by `USERS_DB_PATH` (HTTP transport). Key invariants:

- **The registry (`users.db`) is owned and written by CLORAG** (its `/admin/users` page); this server is a pure reader. `users_db.py` opens SQLite with `mode=ro` so the process can never write even if the volume mount is rw. The schema contract (`users`, `api_keys`, `user_odoo_credentials`, `user_skills`) is documented in the `users_db.py` docstring — **never change it here**; it mirrors clorag `core/user_db.py`.
- **Crypto contract**: `token_crypto.py` ports CLORAG's `utils/token_encryption.py` — Fernet key via PBKDF2HMAC-SHA256, **480 000 iterations**, salt from the `.token_salt` file next to the db, password from `TOKEN_ENCRYPTION_KEY` (or `TOKEN_ENCRYPTION_KEY_FILE` Docker secret). Both containers must share the same secret + salt. Changing iterations or salt location breaks decryption of every stored credential.
- **Roles** (from `users.role`): `admin` → unrestricted; `readonly` → `read`-only scopes and **safety classifies every non-safe method as BLOCKED** (`safety.py`, "Read-only profile" branch); other roles (e.g. `support`) → normal safety rules with their personal Odoo account. The static `MCP_API_KEY` maps to the synthetic `env-admin` identity (admin role, env Odoo client).
- **Per-user Odoo clients** (`user_clients.py`): cached per user with a 300s TTL; on expiry the registry's `updated_at` is re-checked so credential rotation propagates without restart while the `requests.Session` is reused when unchanged. A registry user without stored Odoo credentials gets a `PermissionError` with an actionable message (ask admin → `/admin/users`).
- **Skill visibility** (`skill_visibility.py` middleware): `cyanview-*` prompts are filtered in `list_prompts` and re-enforced in `get_prompt` against the user's `user_skills` allowlist. Generic prompts stay visible to everyone. STDIO and admin/static identities see everything; a missing access token in HTTP mode **fails closed** (no cyanview prompts).
- **Startup preflight** (`__main__.py`): HTTP mode exits 1 unless `USERS_DB_PATH` or `MCP_API_KEY` is set; with `USERS_DB_PATH` it also verifies the db file, `.token_salt`, and `TOKEN_ENCRYPTION_KEY(_FILE)` exist before serving.
- **Env Odoo creds become optional** in multi-user mode — `app_lifespan` tolerates their absence (per-user clients are built lazily); without `USERS_DB_PATH` a missing env config still fails startup.
- Deployment: `docker-compose.multiuser.yml` overlay mounts the CLORAG data dir read-only at `/registry` and injects the encryption key as a Docker secret.

## Safety layer (v1.10.0 + v1.14.0 token gate)

| Level | Behavior |
|-------|----------|
| `SAFE` | Execute immediately |
| `MEDIUM` | Confirm in `strict` mode (default); only HIGH/BLOCKED in `permissive` |
| `HIGH` | Always confirm |
| `BLOCKED` | Always refuse |

Classification also takes a `role` (multi-user mode): `readonly` users get BLOCKED for any non-safe method, before model rules are even consulted. `tests/test_safety_role.py` pins this.

- **Batch rule (strict mode)**: a MEDIUM method affecting more than one record is escalated to confirmation. The count comes from `_estimate_record_count`, which reads the payload from **either** `args_json` or `kwargs_json` (`_COUNTED_ARG` in `safety.py`: `write`/`unlink`/`copy` → `ids`, `create` → `vals_list`, `load` → `data`; `action_*`/`button_*` → `ids`). v2 is named-args-only, so both spellings must be counted — counting only the positional form lets a bulk `unlink` slip past the gate by moving `ids` into `kwargs_json`.
- **Unknown methods** classify as MEDIUM: confirm in `strict`, allow in `permissive`.

- **BLOCKED_MODELS** (writes always refused): `ir.rule`, `ir.model.access`, `ir.module.module`, `ir.config_parameter`, `ir.model`, `res.users`, `res.groups`, `res.users.apikeys`. The `resolve_json` parameter also rejects these as targets — agents cannot use it to read security-critical data. `res.users.apikeys` is blocked because Odoo 19.1+ exposes programmatic API-key management (`res.users.apikeys.generate` / `.revoke` over JSON-2 — Odoo restricts it to *Settings* admins by default, opt-in for others via `base.enable_programmatic_api_keys`, but the connected API user is often privileged); it's a distinct model name from `res.users`, so without an explicit entry an agent could mint a persistent API key (up to 3 months) that outlives the MCP session.
- **SENSITIVE_MODELS** (writes always confirm, both modes): `account.move`, `account.payment`, `account.bank.statement`, `hr.payslip`, `ir.cron`, `ir.model.fields`. The last enables Studio-style custom-field creation/editing — allowed but always token-gated; whole-model changes (`ir.model`) remain BLOCKED.
- **Cascade warnings** are surfaced for: `sale.order.action_confirm` (creates deliveries), `account.move.action_post` (irreversible journal entries), `stock.picking.button_validate` (stock changes), `purchase.order.button_confirm` (incoming receipts), `account.payment.action_post` (journal + reconciliation).
- **Token gate** (v1.14.0): `_issue_confirmation_token()` issues a single-use, 120s-TTL nonce bound to `(model, method, payload_digest)` — a SHA-256 over the deterministic JSON of the operation payload. The confirmation re-call must reproduce the *exact same args/kwargs* the gate saw at issue time, so an agent can't get a token for `unlink([1])` and then re-call with `unlink([1,2,…,1000])`. Digest covers `{"args": args, "kwargs": kwargs}` post-`resolve_json` and post-context-merge for `execute_method`, the full ops list for `batch_execute`, and the params dict for `execute_workflow`.

```python
# Step 1: triggers gate, returns confirmation_token in hint
result = execute_method("sale.order", "action_confirm", args_json='[[15]]')

# Step 2: confirm with token from step 1
execute_method("sale.order", "action_confirm", args_json='[[15]]',
    confirmed=True, confirmation_token='ymzyOtsZTDTJpKKysu_xWQ')
```

### Locked mode (v1.16.0)

`MCP_SAFETY_MODE=locked` adds four overlapping protections on top of the existing classifier and token gate:

| Protection | Default under `locked` | Override |
|---|---|---|
| Global write kill-switch | on | `MCP_READ_ONLY=false` |
| Side-effect allowlist enforced | on (empty list = no writes) | populate `MCP_WRITE_ALLOWLIST` |
| HTTP bind | `127.0.0.1` | `MCP_HOST=0.0.0.0` |
| Live `fields_get` payload pre-flight | on | `MCP_VALIDATE_PAYLOADS=false` |

Each protection is independently overridable. Read `odoo://server-status` to see the resolved profile at runtime, including any foot-gun warnings.

Three helpers are the single sources of truth — reuse them rather than re-deriving the rules:
- `safety_profile.resolve(env)` / `get_profile()` → the frozen `ResolvedProfile`. The umbrella mode picks defaults; each of the four env vars overrides its own field. Unparseable booleans and malformed allowlist entries are warned about and dropped, never fatal.
- `safety.is_side_effect_method(method)` → **fail-closed** `method not in SAFE_METHODS`. Shared by the read-only guard, the allowlist check, and the payload pre-flight. Do not add a name-shape heuristic (`write`/`action_*`/`button_*`) beside it — Odoo write methods like `message_post`, `toggle_active`, `convert_opportunity` and `create_from_urls` match no such shape, which is exactly why the predicate is a set complement.
- `safety.validate_payload_against_schema(...)` → live `fields_get` pre-flight (via `utils.get_fields_for_model`, 60s TTL) run *before* a confirmation token is issued, so a typo'd field name fails without burning a gate round-trip.

`locked` inherits `strict`'s classifier semantics (`safety._STRICT_EQUIV`) and adds its gates on top — it is not a separate classification path.

**Allowlist syntax**: `MCP_WRITE_ALLOWLIST="sale.order.action_confirm,res.partner.message_post,product.product.*"`. The wildcard matches any method on the named model. There is no `*.method` form (too easy to over-grant).

Audit log via `logging.getLogger("odoo_mcp.safety")` (configured in `__init__.py` to stderr at INFO+) when `MCP_SAFETY_AUDIT=true`. `MCP_SAFETY_MODE`, `MCP_SAFETY_AUDIT`, and `MCP_DEFAULT_CONTEXT` are read at call time, so reconfiguring after import (or `patch.dict(os.environ, …)` in tests) takes effect immediately.

## Configuration

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ODOO_URL` / `ODOO_DB` / `ODOO_USERNAME` / `ODOO_API_KEY` | Yes (optional in multi-user mode) | — | Odoo connection (API key preferred over `ODOO_PASSWORD`). With `USERS_DB_PATH`, only the env-admin fallback uses these; per-user clients still need `ODOO_URL`/`ODOO_DB`. |
| `ODOO_TIMEOUT` | No | `30` | Request timeout (seconds) |
| `ODOO_VERIFY_SSL` | No | `true` | Set `false` to disable cert check (visible startup warning) |
| `ODOO_CONFIG_DIR` | No | — | Extra directory searched **first** for a `.env`, ahead of `./.env` then `~/.config/odoo/.env` (`odoo_client.load_config`) |
| `MCP_TRANSPORT` | No | `stdio` | Or `streamable-http` |
| `MCP_API_KEY` | HTTP: this **or** `USERS_DB_PATH` | — | Static bearer token (single-user HTTP, or admin fallback in multi-user mode). HTTP `sys.exit(1)` if neither is set. |
| `USERS_DB_PATH` | No | — | Path to the CLORAG registry (`users.db`) → enables multi-user mode |
| `TOKEN_ENCRYPTION_KEY` / `TOKEN_ENCRYPTION_KEY_FILE` | With `USERS_DB_PATH` | — | Secret to decrypt registry Odoo credentials — must equal CLORAG's value |
| `MCP_HOST` / `MCP_PORT` | No | `0.0.0.0` / `8080` | HTTP bind |
| `MCP_VERBOSE` | No | `true` | Slant-ASCII startup banner (version, transport, masked creds, safety mode, capability counts). Writes to **stderr** only, so STDIO's stdout stays protocol-clean. Set `false` to silence. |
| `MCP_SAFETY_MODE` | No | `strict` | `permissive`, `strict`, or **`locked`** (new in v1.16.0). `locked` activates `MCP_READ_ONLY=true`, `MCP_WRITE_ALLOWLIST` enforcement, `MCP_HOST=127.0.0.1` default, and `MCP_VALIDATE_PAYLOADS=true`. |
| `MCP_SAFETY_AUDIT` | No | — | `true` to log audit entries to stderr |
| `MCP_DEFAULT_CONTEXT` | No | — | JSON merged into all op contexts. Max 4KB. e.g. `{"lang":"fr_FR"}` |
| `MCP_BOOTSTRAP_MODELS` | No | `res.partner,sale.order,account.move,product.product,stock.picking` | Models for `odoo://session-bootstrap`. Max 20. |
| `MCP_READ_ONLY` | No | derived from mode | `true` to reject every method that is not a known read. Fail-closed: the gate is `method not in SAFE_METHODS`, so unrecognised methods (`message_post`, `toggle_active`, `convert_opportunity`, `create_from_urls`, …) are blocked rather than assumed safe. The 13 SAFE_METHODS (`search_read`, `read`, `fields_get`, …) pass through. |
| `MCP_WRITE_ALLOWLIST` | No | empty | Comma-separated `model.method` (or `model.*`) entries permitted as side effects. Enforced under `locked`, or explicitly via this var. |
| `MCP_VALIDATE_PAYLOADS` | No | derived from mode | `true` to validate write payloads against live `fields_get` before issuing a confirmation token. |

## Tools (5)

| Tool | Purpose |
|------|---------|
| `execute_method` | Universal Odoo API access |
| `batch_execute` | Multiple ops with progress tracking. **`atomic=True` stops at the first failure — it does not roll back.** Each op is a separate JSON-2 request, so completed ops stay committed |
| `execute_workflow` | Two implemented workflows: `lead_to_won` (aliases `crm_workflow`, `opportunity_won`) and `create_and_post_invoice` (alias `quick_invoice`). Anything else returns `Unknown workflow` — there is **no** natural-language workflow synthesis. Note `safety._WORKFLOW_STEPS` still carries a `stock_transfer` entry with no matching branch in `execute_workflow`; it classifies, then fails as unknown |
| `configure_odoo` | Interactive connection setup (user elicitation) |
| `read_resource` | Read any `odoo://` URI — bridge for clients without resource template support |

## Discovery resources (selected — full list at `odoo://templates`)

| Resource | When to use |
|----------|-------------|
| `odoo://model/{model}/quick-schema` | **Default for schema introspection.** ~1.5 KB, short keys (`t`/`req`/`ro`/`rel`), no labels |
| `odoo://model/{model}/fields` | Lightweight (~5–10 KB) with labels |
| `odoo://model/{model}/schema` | Full schema with relationships (~300 KB) |
| `odoo://model/{model}/workflow` | State machine transitions, methods, side effects, irreversible flags. 6 main models hardcoded; dynamic fallback for others |
| `odoo://bundle/{m1,m2,...}` | Batch quick-schema, max 10 models |
| `odoo://session-bootstrap` | One-call kickoff: schemas + workflows for `MCP_BOOTSTRAP_MODELS` |
| `odoo://methods/{model}` | Live-enriched (signatures, return types, decorators) via `/doc-bearer/`; static fallback |
| `odoo://model-limitations` | Known issues + runtime-detected problematic combos |
| `odoo://domain-syntax` / `odoo://aggregation` / `odoo://pagination` / `odoo://hierarchical` | Reference docs |
| `odoo://server-status` | Resolved safety profile, transport, host, allowlist, warnings. Non-secret. New in v1.16.0. |

When read through the `read_resource` tool (rather than a native resource client), every one of these is capped at 15 000 chars by default — `odoo://model/{model}/schema` needs `max_chars=0` to come back whole.

## Key conventions

**Schema first, query second.** Never guess field names — read `odoo://model/{model}/quick-schema` first. Guessing wastes API calls; introspection is fast.

**`args_json` / `kwargs_json` are JSON strings, not Python objects.** Pass `args_json='[[15]]'`, not `args_json=[[15]]`. Same for `kwargs_json` and `resolve_json`. The server parses them with `json.loads`; native lists/dicts will fail validation.

**`@api.private` is enforced.** Methods like `check_access` (use `has_access`) and `search_fetch` (use `search_read`) are blocked before the API call with actionable hints. Methods starting with `_` are also checked dynamically against `/doc-bearer/`.

**Match the method to the question.** Counting → `search_count` (returns just an int — no payload, no pagination limit silently capping the result). Grouped counts / sums / averages → `formatted_read_group`. Only use `search_read` when you actually need the records. See `odoo://aggregation`.

**`read_group` is deprecated in v19.** Use `formatted_read_group` (param is `aggregates`, not `fields`).

**Domain logic is Polish-prefix.** `["&", t1, t2]` AND, `["|", t1, t2]` OR, `["!", t]` NOT. Dot notation works (`["partner_id.country_id.code", "=", "US"]`) but can break on computed fields — check `odoo://model-limitations`.

**One2many / Many2many command tuples**: `(0,0,vals)` create, `(1,id,vals)` update, `(2,id,0)` delete, `(4,id,0)` link, `(6,0,[ids])` replace all.

**Many2one resolution.** Use `resolve_json` to pass names instead of IDs:

```python
execute_method("res.partner", "write",
    args_json='[[1], {"user_id": null}]',
    resolve_json='{"user_id": {"model": "res.users", "search": "Administrator"}}')
```

Returns options on ambiguous match.

## Known model limitations

`stock.move.line`:
- `picking_type_id` with `!=` returns `NotImplemented` (computed field) → use `picking_id.picking_type_id`
- `lots_visible` is non-stored computed → exclude from `fields`, fetch separately
- `product_category_name` is a 3-level deep related → exclude from `fields`/`domain`
- Dot notation filters can fail on complex JOINs → query related models separately

Read `odoo://model-limitations` for the full live list (static + runtime-detected).

## Security hardening (v1.13.0 — keep this enforced)

- **Input validation**: model `^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$` (max 128); method `^[a-zA-Z_][a-zA-Z0-9_]*$` (max 64); URIs must be `odoo://`.
- **Thread safety**: `OdooClient` is a singleton with double-checked locking. `_DOC_CACHE` (100-entry LRU), `_FIELDS_CACHE` (100-entry LRU), and `RUNTIME_MODEL_ISSUES` each use their own `threading.Lock`.
- **Error sanitization**: never include Odoo `debug` tracebacks in MCP responses.
- **Docker**: non-root user (UID 1001); `run-docker.sh` uses `--env-file`; `docker-compose.yml` uses `${MCP_API_KEY:?required}`.
- **Docker must not restate dependency constraints.** The Dockerfile installs a stub package to resolve `pyproject.toml`'s dependency set into its own cached layer, then installs the real source with `--no-deps`. A hand-copied list drifts silently — it had already lost the `fastmcp<4` ceiling and never listed `cryptography>=42`. `tests/test_dependency_pins.py::TestDockerfileUsesDeclaredDependencies` fails if a restated pin reappears.
- **HTTP**: `sys.exit(1)` unless `MCP_API_KEY` or `USERS_DB_PATH` is set. Wizard auto-generates keys with `secrets.token_urlsafe(32)`. Multi-user token compare uses `hmac.compare_digest` (static key) / sha256 hash lookup (registry keys — plaintext keys are never stored).
- **Resource limits**: `MCP_DEFAULT_CONTEXT` ≤ 4KB; `MCP_BOOTSTRAP_MODELS` ≤ 20 models; `_DOC_CACHE` and `_FIELDS_CACHE` ≤ 100 entries each.
- **Gitignored**: `.env`, `.env.local`, `.mcp.json`, `odoo_config.json`.

## Release process

A release touches four places — keep them in sync:

1. Version string in **both** `pyproject.toml` and `src/odoo_mcp/__init__.py` (`__version__`).
2. `CHANGELOG.md` — Keep a Changelog format; move entries from `[Unreleased]` into the new version section with the date.
3. This file's header (version + surface counts) if they changed.
4. `wiki/` — a **gitignored local clone of the GitHub wiki** (`AlanOgic/odoo-mcp-19.wiki.git`). Wiki pages (Tools, Resources, Prompts, Deployment, …) are refreshed at each release and need their **own commit and push inside `wiki/`** — committing this repo does not publish them.

Release commit convention: `chore(release): X.Y.Z — <summary>`.

**No public registries — by choice.** The package is deliberately not published to PyPI, and no Docker Hub image exists. Distribution is `pip install git+https://github.com/AlanOgic/odoo-mcp-19.git` or a locally built Docker image (`docker build -t odoo-mcp-19 .`). Do not add publish steps, and do not treat the missing PyPI/Docker Hub listings as a bug to fix.

## Notes for Claude Code

- This is a **v2-only** server. Do not add v1 fallback code.
- `module_knowledge.json`, `assets/*.svg` (the brand icon `app.py` loads), and `skills/*.md` (the cyanview prompt bodies) must remain in `[tool.setuptools.package-data]` so they ship in the wheel and Docker image.
- The registry schema (`users_db.py` docstring) and crypto parameters (`token_crypto.py`, 480k PBKDF2 iterations) are **contracts owned by CLORAG** (the registry-managing app, separate repo at `~/dev/clorag` — schema source `core/user_db.py`, crypto source `utils/token_encryption.py`) — changes must happen there first; this repo only mirrors them. `tests/conftest.py` re-implements both contracts to seed test registries; keep it in sync.
- `skills/*.md` are copies of the curated `~/.claude/skills/cyanview-*` sources with frontmatter intact (stripped at load by `skill_prompts.load_skill`). When updating a skill, update the source and re-copy.
- `arg_mapping.py` is mandatory — v2 API rejects positional args. Adding a new ORM method = entry in `arg_mapping`. **Record-bound methods take their recordset in the JSON-2 body `ids` key**: all `action_*`/`button_*` entries map position 0 → `"ids"` (so does `copy`), and `convert_args_to_v2` has a generic fallback that routes a leading list-of-ints to `ids` for unmapped record-bound methods (e.g. `action_set_won`) instead of dropping it. `tests/test_arg_mapping.py` pins this contract.
- **Every positional parameter a method accepts must appear in its `V2_ARG_MAPPING` entry.** `convert_args_to_v2` iterates the supplied args, not the table, and raises `ValueError` on a positional it has no name for — JSON-2 is named-args-only, so an unmappable positional cannot be forwarded, and dropping it silently sends a *different but still valid* request (a `name_search` minus its domain matched everything; a `search` minus its limit returned the server default). Transcribe entries from the Odoo source, **not the docstrings** — `default_get`'s still says `fields_list` while the parameter has been `fields` (`odoo/orm/models.py` @ 19.0), which made every `default_get` call 422. `read_group` is the one method whose sort parameter really is `orderby`, so it is exempt from the `V2_KWARGS_MAPPING` `orderby`→`order` rename via `_KWARGS_MAPPING_EXEMPT`.
- `odoo_client.py` always sends `Authorization: Bearer` and returns the JSON-2 response body as-is — **no `{"result": ...}` envelope unwrap** (that was the legacy `/jsonrpc` convention; unwrapping would corrupt methods that legitimately return a dict with a `result` key). `tests/test_odoo_client.py` pins this.
- The canonical MCP server name in client configs (README, setup wizard, Claude Desktop config) is **`odoo19-mcp`** — keep it consistent when touching docs or the wizard.
- `live` tests are not pytest-collected; they are direct scripts that mutate env state. Don't reorganize them into pytest fixtures without checking the in-file note.
- AI module (Enterprise): models `ai.agent`, `ai.topic`, `ai.agent.source`, `ai.embedding`. Special methods: `get_direct_response`, `create_from_urls`, `create_from_attachments`. Documented in `module_knowledge.json`.
