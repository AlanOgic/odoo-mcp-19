# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**odoo-mcp-19** — Standalone MCP server for Odoo 19+ using the **v2 JSON-2 API** (`POST /json/2/{model}/{method}`, Bearer token auth, named args only). No v1 fallback.

- **Version**: 1.14.0 · **Python**: 3.10+ · **MCP**: 2025-11-25 (FastMCP 3.2.0+)
- **Surface**: 5 tools, 27 `odoo://` resources, 13 prompts
- **Discovery is via resources, action is via tools** — there is no `list_models` tool, agents read `odoo://models` instead.

## Development commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Install published package (production path — also installs the `odoo-mcp-19` console entry point)
pip install odoo-mcp-19

# Run server — STDIO (default); loads .env from cwd
python -m odoo_mcp

# Run server — HTTP (requires MCP_API_KEY)
MCP_TRANSPORT=streamable-http MCP_API_KEY=<token> python -m odoo_mcp

# Interactive setup wizard — generates .env, Docker cmd, Claude Desktop config
python -m odoo_mcp --setup

# Tests — unit (no Odoo needed)
pytest tests/test_safety.py tests/test_token_gate.py tests/test_resources.py
pytest tests/test_safety.py::TestClassifyOperation::test_safe_methods_are_safe   # single test

# Tests — live (requires .env with real Odoo creds; script-style runners)
python tests/live/test_safety_live.py
python tests/live/test_v1110_live.py

# Format + lint + typecheck
black . && isort .
ruff check .
mypy src/odoo_mcp

# Docker
docker build -t odoo-mcp-19 .
docker compose up -d           # uses .env, requires MCP_API_KEY
```

Note: live tests under `tests/live/` are **script-style runners**, not pytest modules — invoke them directly with `python`. They mutate environment state.

- **Unit (no Odoo)**: `tests/test_safety.py`, `tests/test_token_gate.py`, `tests/test_resources.py` (patches `get_odoo_client` with a stub — pins resource-layer validation/error handling) — run with `pytest`.
- **Live (need `.env`)**: anything under `tests/live/` — run with `python <file>`, not pytest.

## High-level architecture

The package was split in v1.14.0 (commit `ea10d79`) from a 3762-line `server.py` into 12 focused modules. Import order matters: `app.py` must be imported first so the `mcp` decorator is bound before `server.py`, `resources.py`, and `prompts.py` register their handlers.

```
src/odoo_mcp/
├── __main__.py        CLI entry: STDIO/HTTP bootstrap, --setup wizard, MCP_API_KEY enforcement
├── app.py             FastMCP instance + Odoo brand icon — imported first
├── server.py          5 tools + _RESOURCE_ROUTES table + search_read fallback + safety integration
├── resources.py       27 odoo:// resource handlers
├── prompts.py         13 guided prompts
├── safety.py          Risk classification + token gate (v1.14.0)
├── odoo_client.py     v2 JSON-2 client (thread-safe singleton, sanitized errors)
├── arg_mapping.py     Positional → named args for 30 ORM methods
├── constants.py       Limits, regex validators, MODEL_STATE_MACHINES, default context
├── models.py          Pydantic response schemas (structured output)
├── utils.py           Compact schema builder, error suggestions, /doc-bearer LRU cache
└── module_knowledge.json   Special methods for 13 modules (loaded at startup, shipped as package data)
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

**2. Resource bridge** — `read_resource(uri)` exists because some clients (Claude Desktop) don't speak resource templates. The `_RESOURCE_ROUTES` table in `server.py` maps URIs to the same handlers `resources.py` registers, so the same `odoo://...` URI works either way.

**3. Live doc enrichment** — `odoo://methods/{model}` and `@api.private` detection both consult `/doc-bearer/<model>.json` (provided by Odoo's `api_doc` module, requires `api_doc.group_allow_doc` on the API user). Cached in `_DOC_CACHE`: 5-min TTL, 100-entry LRU, `threading.Lock`. Falls back silently to static data if unavailable.

**4. Background tasks** — `batch_execute` and `execute_workflow` use FastMCP's `[tasks]` extra to support async execution with progress reporting (MCP 2025-11-25 SEP-1686).

## Safety layer (v1.10.0 + v1.14.0 token gate)

| Level | Behavior |
|-------|----------|
| `SAFE` | Execute immediately |
| `MEDIUM` | Confirm in `strict` mode (default); only HIGH/BLOCKED in `permissive` |
| `HIGH` | Always confirm |
| `BLOCKED` | Always refuse |

- **BLOCKED_MODELS** (writes always refused): `ir.rule`, `ir.model.access`, `ir.module.module`, `ir.config_parameter`, `ir.model`, `ir.model.fields`, `res.users`, `res.groups`. The `resolve_json` parameter also rejects these as targets — agents cannot use it to read security-critical data.
- **SENSITIVE_MODELS** (writes always confirm): `account.move`, `account.payment`, `account.bank.statement`, `hr.payslip`, `ir.cron`.
- **Cascade warnings** are surfaced for: `sale.order.action_confirm` (creates deliveries), `account.move.action_post` (irreversible journal entries), `stock.picking.button_validate` (stock changes), `purchase.order.button_confirm` (incoming receipts), `account.payment.action_post` (journal + reconciliation).
- **Token gate** (v1.14.0): `_issue_confirmation_token()` issues a single-use, 120s-TTL nonce bound to `(model, method, payload_digest)` — a SHA-256 over the deterministic JSON of the operation payload. The confirmation re-call must reproduce the *exact same args/kwargs* the gate saw at issue time, so an agent can't get a token for `unlink([1])` and then re-call with `unlink([1,2,…,1000])`. Digest covers `{"args": args, "kwargs": kwargs}` post-`resolve_json` and post-context-merge for `execute_method`, the full ops list for `batch_execute`, and the params dict for `execute_workflow`.

```python
# Step 1: triggers gate, returns confirmation_token in hint
result = execute_method("sale.order", "action_confirm", args_json='[[15]]')

# Step 2: confirm with token from step 1
execute_method("sale.order", "action_confirm", args_json='[[15]]',
    confirmed=True, confirmation_token='ymzyOtsZTDTJpKKysu_xWQ')
```

Audit log via `logging.getLogger("odoo_mcp.safety")` (configured in `__init__.py` to stderr at INFO+) when `MCP_SAFETY_AUDIT=true`. `MCP_SAFETY_MODE`, `MCP_SAFETY_AUDIT`, and `MCP_DEFAULT_CONTEXT` are read at call time, so reconfiguring after import (or `patch.dict(os.environ, …)` in tests) takes effect immediately.

## Configuration

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ODOO_URL` / `ODOO_DB` / `ODOO_USERNAME` / `ODOO_API_KEY` | Yes | — | Odoo connection (API key preferred over `ODOO_PASSWORD`) |
| `ODOO_TIMEOUT` | No | `30` | Request timeout (seconds) |
| `ODOO_VERIFY_SSL` | No | `true` | Set `false` to disable cert check (visible startup warning) |
| `MCP_TRANSPORT` | No | `stdio` | Or `streamable-http` |
| `MCP_API_KEY` | **Yes for HTTP** | — | Bearer token. Server `sys.exit(1)` without it in HTTP mode. |
| `MCP_HOST` / `MCP_PORT` | No | `0.0.0.0` / `8080` | HTTP bind |
| `MCP_VERBOSE` | No | `true` | Slant-ASCII startup banner (version, transport, masked creds, safety mode, capability counts). Writes to **stderr** only, so STDIO's stdout stays protocol-clean. Set `false` to silence. |
| `MCP_SAFETY_MODE` | No | `strict` | Or `permissive` (only HIGH/BLOCKED confirm) |
| `MCP_SAFETY_AUDIT` | No | — | `true` to log audit entries to stderr |
| `MCP_DEFAULT_CONTEXT` | No | — | JSON merged into all op contexts. Max 4KB. e.g. `{"lang":"fr_FR"}` |
| `MCP_BOOTSTRAP_MODELS` | No | `res.partner,sale.order,account.move,product.product,stock.picking` | Models for `odoo://session-bootstrap`. Max 20. |

## Tools (5)

| Tool | Purpose |
|------|---------|
| `execute_method` | Universal Odoo API access |
| `batch_execute` | Multiple ops with progress tracking |
| `execute_workflow` | Pre-built multi-step workflows (`quote_to_cash`, `lead_to_won`, `create_and_post_invoice`, …) |
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

## Key conventions

**Schema first, query second.** Never guess field names — read `odoo://model/{model}/quick-schema` first. Guessing wastes API calls; introspection is fast.

**`args_json` / `kwargs_json` are JSON strings, not Python objects.** Pass `args_json='[[15]]'`, not `args_json=[[15]]`. Same for `kwargs_json` and `resolve_json`. The server parses them with `json.loads`; native lists/dicts will fail validation.

**`@api.private` is enforced.** Methods like `check_access` (use `has_access`) and `search_fetch` (use `search_read`) are blocked before the API call with actionable hints. Methods starting with `_` are also checked dynamically against `/doc-bearer/`.

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
- **Thread safety**: `OdooClient` is a singleton with double-checked locking. `_DOC_CACHE` (100-entry LRU) and `RUNTIME_MODEL_ISSUES` use `threading.Lock`.
- **Error sanitization**: never include Odoo `debug` tracebacks in MCP responses.
- **Docker**: non-root user (UID 1001); `run-docker.sh` uses `--env-file`; `docker-compose.yml` uses `${MCP_API_KEY:?required}`.
- **HTTP**: `sys.exit(1)` if `MCP_API_KEY` unset. Wizard auto-generates with `secrets.token_urlsafe(32)`.
- **Resource limits**: `MCP_DEFAULT_CONTEXT` ≤ 4KB; `MCP_BOOTSTRAP_MODELS` ≤ 20 models; `_DOC_CACHE` ≤ 100 entries.
- **Gitignored**: `.env`, `.env.local`, `.mcp.json`, `odoo_config.json`.

## Notes for Claude Code

- This is a **v2-only** server. Do not add v1 fallback code.
- `module_knowledge.json` must remain in `[tool.setuptools.package-data]` so it ships in the wheel and Docker image.
- `arg_mapping.py` is mandatory — v2 API rejects positional args. Adding a new ORM method = entry in `arg_mapping`.
- `live` tests are not pytest-collected; they are direct scripts that mutate env state. Don't reorganize them into pytest fixtures without checking the in-file note.
- AI module (Enterprise): models `ai.agent`, `ai.topic`, `ai.agent.source`, `ai.embedding`. Special methods: `get_direct_response`, `create_from_urls`, `create_from_attachments`. Documented in `module_knowledge.json`.
