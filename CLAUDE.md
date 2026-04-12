# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**Odoo MCP Server 19** - A standalone MCP server for Odoo 19+ using the v2 JSON-2 API.

- **Version**: 1.14.0
- **MCP Spec**: MCP 2025-11-25 (FastMCP 3.2.0+)
- **Odoo Support**: v19+ only (v2 JSON-2 API)

### MCP 2025-11-25 Features (v1.9.0)

- **Background Tasks** (SEP-1686): `batch_execute` and `execute_workflow` support async execution with progress tracking
- **Icons** (SEP-973): Server and tool icons using Odoo brand colors (#714B67)
- **Structured Output Schemas**: Typed Pydantic response models for all tools
- **User Elicitation**: Interactive `configure_odoo` tool for connection setup

## Architecture

### Package Structure

```
odoo-mcp-19/
├── src/odoo_mcp/
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # CLI entry point (STDIO / HTTP bootstrap, setup wizard)
│   ├── app.py                   # FastMCP instance + icon (imported first for tool registration)
│   ├── server.py                # MCP tool implementations (execute_method, batch_execute, ...)
│   ├── resources.py             # odoo:// resource handlers (schema, workflow, bundle, ...)
│   ├── prompts.py               # Guided MCP prompts (13 workflows)
│   ├── safety.py                # Safety classification engine (v1.10.0) + token gate (v1.14.0)
│   ├── odoo_client.py           # Odoo v2 JSON-2 API client (thread-safe singleton)
│   ├── arg_mapping.py           # Positional → named args conversion (30 ORM methods)
│   ├── constants.py             # Limits, validators, state machines, default context
│   ├── models.py                # Pydantic response schemas (structured output)
│   ├── utils.py                 # Compact schema builder, error suggestions, /doc-bearer cache
│   ├── module_knowledge.json    # Module-specific methods knowledge base
│   └── assets/
│       └── odoo_icon.svg        # Odoo brand icon for MCP clients
├── tests/
│   ├── test_safety.py           # Unit tests for safety layer
│   └── live/                    # Live integration tests (require .env + real Odoo)
├── Dockerfile                   # Docker build (non-root user)
├── docker-compose.yml           # Docker compose config (mandatory MCP_API_KEY)
├── run-docker.sh                # Docker wrapper for Claude Desktop (--env-file)
├── pyproject.toml               # Package configuration
└── README.md                    # User documentation
```

### Key Components

**1. MCP Server Core** — the former 3762-line `server.py` was split in v1.14.0 (commit `ea10d79`) into focused modules. Each module has one responsibility:

- **`app.py`** (~90 lines) — creates the `FastMCP` instance and registers the Odoo brand icon. Imported first so that the `mcp` decorator is available when `server.py`, `resources.py`, and `prompts.py` register their handlers.
- **`server.py`** (~1226 lines) — implements the **5 tools**: `execute_method`, `batch_execute`, `execute_workflow`, `configure_odoo`, `read_resource`. Contains the `_RESOURCE_ROUTES` table for clients that don't speak resource templates, automatic `search_read → search+read` fallback, runtime issue tracking, safety-layer integration, and background-task progress reporting.
- **`resources.py`** (~1244 lines) — implements all **27 `odoo://` resources**: models/schema/quick-schema/fields/methods/workflow/bundle/session-bootstrap/actions/tools/domain-syntax/model-limitations/templates.
- **`prompts.py`** (~443 lines) — registers the **13 guided prompts** (odoo-exploration, search-records, quote-to-cash, crm-pipeline, customer-360, etc.).
- **`constants.py`** (~471 lines) — limits (`DEFAULT_LIMIT=100`, `MAX_LIMIT=1000`), regex validators (`_validate_model`, `_validate_method`), `MODEL_STATE_MACHINES`, `PRIVATE_METHOD_HINTS`, and the `MCP_DEFAULT_CONTEXT` loader + `_merge_context` helper.
- **`models.py`** (~83 lines) — Pydantic response schemas: `ExecuteMethodResponse`, `BatchExecuteResponse`, `BatchOperationResult`, `ExecuteWorkflowResponse`, `WorkflowStepResult`, `IssueAnalysis`.
- **`utils.py`** (~421 lines) — `_build_compact_schema`, `get_error_suggestion` (~25 expanded error patterns with `{model}` templating), `_get_live_doc` + thread-safe `_DOC_CACHE` (100 entries, LRU), `_track_model_issue` + thread-safe `RUNTIME_MODEL_ISSUES`.

**2. Odoo Client** (`odoo_client.py`)
- v2 JSON-2 API only (Bearer token auth)
- Endpoint: `/json/2/{model}/{method}`
- Thread-safe singleton (`get_odoo_client()` with double-checked locking)
- Automatic argument conversion via arg_mapping
- Live model documentation via `/doc-bearer/<model>.json` (api_doc module)
- Server tracebacks logged to stderr only — never forwarded to MCP clients
- SSL disable warning on startup when `ODOO_VERIFY_SSL=false` + HTTPS

**3. Safety Layer** (`safety.py`, ~501 lines)
- Pre-execution risk classification: `SAFE` / `MEDIUM` / `HIGH` / `BLOCKED`
- `BLOCKED_MODELS`: `ir.rule`, `ir.model.access`, `ir.module.module`, `ir.config_parameter`, `ir.model`, `ir.model.fields`, `res.users`, `res.groups` (writes always refused)
- `SENSITIVE_MODELS`: `account.move`, `account.payment`, `account.bank.statement`, `hr.payslip`, `ir.cron` (writes always confirm)
- Cascade warnings for `action_confirm`, `action_post`, `button_validate`, `button_confirm`
- **v1.14.0 token-based confirmation**: `_issue_confirmation_token()` creates a single-use, time-limited (120s), operation-bound cryptographic nonce. Agents cannot bypass the gate by passing `confirmed=true` — they must present a valid token from the gate response.
- `classify_operation`, `classify_batch`, `classify_workflow`, `audit_log` (to stderr when `MCP_SAFETY_AUDIT=true`)

**4. Argument Mapping** (`arg_mapping.py`)
- Converts positional args to named args for v2 API
- Supports 30 ORM methods (search, create, write, formatted_read_group, has_access, action_*, button_*, etc.)

**5. Module Knowledge** (`module_knowledge.json`)
- Special methods for 13 Odoo modules (including AI module)
- 30 ORM methods documented
- Error patterns with suggestions
- Validated against Odoo 19 source code

**6. AI Module** (Enterprise only)
- Models: ai.agent, ai.topic, ai.agent.source, ai.embedding
- LLM support: OpenAI (gpt-4o, gpt-5), Google (gemini-2.5-pro/flash)
- RAG system with pgvector embeddings
- Special methods: get_direct_response, create_from_urls, create_from_attachments

## Development Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run server (STDIO)
python -m odoo_mcp

# Build Docker
docker build -t odoo-mcp-19 .

# Test Docker
docker run --rm -i \
  -e ODOO_URL=https://your.odoo.com \
  -e ODOO_DB=db \
  -e ODOO_USERNAME=user \
  -e ODOO_API_KEY=key \
  odoo-mcp-19

# Format code
black . && isort .

# Lint
ruff check .
```

## Configuration

Required environment variables:
- `ODOO_URL` - Odoo server URL
- `ODOO_DB` - Database name
- `ODOO_USERNAME` - Login username
- `ODOO_API_KEY` - API key (recommended over password)

Optional:
- `ODOO_PASSWORD` - Password (fallback)
- `ODOO_TIMEOUT` - Request timeout (default: 30)
- `ODOO_VERIFY_SSL` - SSL verification (default: true)

HTTP Transport (v1.8.0+):
- `MCP_TRANSPORT` - Transport mode: `stdio` (default) or `streamable-http`
- `MCP_API_KEY` - Bearer token for HTTP authentication (**required** for streamable-http — server refuses to start without it)
- `MCP_HOST` - HTTP bind address (default: 0.0.0.0)
- `MCP_PORT` - HTTP port (default: 8080)

DX Configuration (v1.11.0+):
- `MCP_DEFAULT_CONTEXT` - JSON object merged into all operation contexts (e.g. `{"lang": "fr_FR"}`). Max 4KB.
- `MCP_BOOTSTRAP_MODELS` - Comma-separated model names for session-bootstrap (default: `res.partner,sale.order,account.move,product.product,stock.picking`). Max 20 models.

## v2 API Details

The v2 JSON-2 API requires:
- Bearer token authentication via `Authorization` header
- Database name via `X-Odoo-Database` header
- Named arguments only (no positional args)

Endpoint format: `POST /json/2/{model}/{method}`

Example request:
```json
{
  "domain": [["is_company", "=", true]],
  "fields": ["name", "email"],
  "limit": 10
}
```

## Module Knowledge System

The `module_knowledge.json` contains:
- Special methods that replace standard ORM methods (e.g., `article_create` instead of `create`)
- Workflow methods for business processes
- Error patterns with contextual suggestions

To add a new module:
1. Add entry to `modules` in `module_knowledge.json`
2. Include `model`, `special_methods`, and optionally `notes`

## Error Handling

The server provides smart error suggestions based on:
- HTTP status codes (422, 403, 404)
- Model and method context
- Known error patterns

Example: A 422 error on `knowledge.article.create` suggests using `article_create(title='...')` instead.

## Automatic Fallback System (v1.7.0+)

When `search_read` fails with a 500 error, the server automatically:

1. **Falls back to `search` + `read`** - Executes them separately
2. **Categorizes the error** - timeout, relational_filter, computed_field, access_rights, memory, data_integrity
3. **Detects problematic patterns** - dot notation, negative operators on computed fields, deep related fields
4. **Tracks issues at runtime** - Builds a knowledge base of problematic model/method combinations
5. **Provides actionable solutions** - Based on error category and detected patterns

### Response with Fallback

```json
{
  "success": true,
  "result": [...],
  "fallback_used": true,
  "issue_analysis": {
    "category": "relational_filter",
    "cause": "Relational/dot notation filter issue",
    "domain_patterns": ["dot_notation"],
    "problematic_fields": ["lots_visible (non_stored_computed)"],
    "suggested_solutions": ["Avoid dot notation in domain", "Query related model separately"],
    "model_specific_advice": ["Safe fields: id, product_id, lot_id, quantity..."]
  }
}
```

### Model Limitations Resource

Read `odoo://model-limitations` to see:
- **Static limitations** - Verified issues from source code analysis (e.g., `stock.move.line`)
- **Runtime detected** - Issues discovered during operation
- **Pattern summary** - Aggregated problematic patterns across all models
- **Recommendations** - Global suggestions based on detected patterns

### Known Limitations: `stock.move.line`

| Issue | Cause | Solution |
|-------|-------|----------|
| `picking_type_id` with `!=` | Computed field returns `NotImplemented` | Use `picking_id.picking_type_id` instead |
| `lots_visible` in fields | Non-stored computed field | Exclude from fields, fetch separately |
| `product_category_name` | 3-level deep related field | Exclude from fields/domain |
| Dot notation filters | Complex JOINs may fail | Query related models separately |

## Safety Layer (v1.10.0)

Pre-execution safety classification that gates dangerous operations behind confirmation.

### Risk Levels

| Level | Behavior | Confirm? |
|-------|----------|----------|
| `SAFE` | Execute immediately | Never |
| `MEDIUM` | Gate based on mode/volume | Conditional |
| `HIGH` | Always require confirmation | Always |
| `BLOCKED` | Always refuse | N/A |

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SAFETY_MODE` | `strict` | `strict` = MEDIUM batch writes confirm; `permissive` = only HIGH/BLOCKED |
| `MCP_SAFETY_AUDIT` | (disabled) | `true` to log audit entries to stderr |

### Confirmation Flow (Token-Based, v1.14.0)

Confirmation uses a cryptographic nonce token to prevent bypass. An agent cannot skip the safety gate by always passing `confirmed=true` — it must present a valid, single-use token from the gate response.

1. Caller sends `execute_method(model, method, args_json, kwargs_json)`
2. Safety layer classifies the operation
3. If confirmation needed → returns `pending_confirmation=true` with `safety` classification and a `confirmation_token` in the `hint` field
4. Caller reviews, then re-calls with `confirmed=true` AND `confirmation_token='<token>'` to proceed

```python
# Step 1: Call triggers safety gate
result = execute_method("sale.order", "action_confirm", args_json='[[15]]')
# → pending_confirmation=true, hint contains token

# Step 2: Confirm with the token from step 1
execute_method("sale.order", "action_confirm", args_json='[[15]]',
    confirmed=true, confirmation_token='ymzyOtsZTDTJpKKysu_xWQ')
```

**Token rules:**
- Single-use: consumed on validation
- Time-limited: expires after 120 seconds
- Operation-bound: tied to the specific model+method from the original classification
- Same flow applies to `batch_execute` and `execute_workflow`

### Blocked Models (write always refused)

`ir.rule`, `ir.model.access`, `ir.module.module`, `ir.config_parameter`, `ir.model`, `ir.model.fields`, `res.users`, `res.groups`

### Sensitive Models (write always confirms)

`account.move`, `account.payment`, `account.bank.statement`, `hr.payslip`, `ir.cron`

### Cascade Warnings

Known side effects are surfaced for:
- `sale.order` + `action_confirm` → creates deliveries
- `account.move` + `action_post` → creates journal entries (irreversible)
- `stock.picking` + `button_validate` → updates stock levels
- `purchase.order` + `button_confirm` → creates incoming receipts
- `account.payment` + `action_post` → creates journal entries + reconciliation

### Integration

Safety checks are integrated in `execute_method`, `batch_execute`, and `execute_workflow`. The `confirmed` parameter requires a `confirmation_token` from the safety gate response (v1.14.0+).

### resolve_json Security

The `resolve_json` parameter validates target models against both `_validate_model()` regex and `BLOCKED_MODELS`. An agent cannot use `resolve_json` to read from security-critical models like `res.users`, `ir.config_parameter`, etc.

## DX Improvements (v1.11.0)

### Quick Schema (`odoo://model/{model}/quick-schema`)

Ultra-compact schema (~1.5-2KB vs 5-10KB for `/fields`). Short keys: `t` (type), `req` (required), `ro` (readonly), `rel` (relation). No labels, no help text, no indentation. Use as **default** for schema introspection.

### Bundle (`odoo://bundle/{models}`)

Batch quick-schema for N models in one call. Max 10 models. Format: `odoo://bundle/res.partner,sale.order,stock.picking`

### Session Bootstrap (`odoo://session-bootstrap`)

One call to bootstrap a conversation: quick-schemas + state machine workflows for common models. Configurable via `MCP_BOOTSTRAP_MODELS` env var. Default: `res.partner,sale.order,account.move,product.product,stock.picking`

### Workflow Resource (`odoo://model/{model}/workflow`)

State machine transitions for 6 main models: `sale.order`, `account.move`, `crm.lead`, `stock.picking`, `purchase.order`, `hr.leave`. Each transition includes: from, to, method, label, side_effects, irreversible flag. Dynamic fallback for unmapped models (reads state field + action methods from live doc).

### Many2one Resolution (`resolve_json` parameter)

New optional parameter on `execute_method` to auto-resolve Many2one field names to IDs:
```python
execute_method("res.partner", "write",
    args_json='[[1], {"user_id": null}]',
    resolve_json='{"user_id": {"model": "res.users", "search": "Administrator"}}')
```
Uses `name_search` internally. Returns error with options if ambiguous (>1 match) or no match.

### Default Context (`MCP_DEFAULT_CONTEXT`)

Set `MCP_DEFAULT_CONTEXT` env var (JSON object) to apply default context to all operations:
```bash
export MCP_DEFAULT_CONTEXT='{"lang": "fr_FR", "tz": "Europe/Paris"}'
```
Explicit context in kwargs takes priority. Applied in `execute_method` and `batch_execute`.

### Expanded Error Patterns

Error patterns expanded from ~5 to ~25 with template variable `{model}` substitution. Covers:
- 422: singleton, null value, invalid field, readonly, Many2one type, unique constraint
- 500: OperationalError, NoneType, NotImplementedError, statement timeout
- 403: ir.rule, group_ security
- 404: json/2 endpoint, doc-bearer
- Fallback patterns: Many2one, singleton, readonly (matched regardless of HTTP status)

### Bug Fix: Context in Fallback

The `context` parameter is now correctly preserved when `search_read` falls back to `search` + `read` on 500 errors.

### New Resources Summary (27 total, +4 new)

| Resource | Description |
|----------|-------------|
| `odoo://model/{name}/quick-schema` | Ultra-compact schema (~1.5KB) |
| `odoo://model/{name}/workflow` | State machine transitions |
| `odoo://bundle/{models}` | Batch quick-schema (max 10) |
| `odoo://session-bootstrap` | Bootstrap conversation |

## Security Hardening (v1.13.0)

### Input Validation
- **Model names**: Regex `^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$`, max 128 chars. Applied in `execute_method` and `batch_execute`.
- **Method names**: Regex `^[a-zA-Z_][a-zA-Z0-9_]*$`, max 64 chars. Blocks path traversal (`../`) and query injection (`?param=`).
- **`read_resource` URI**: Must start with `odoo://`. Blocks `file://`, `http://`, etc.
- **`batch_execute` args**: Each operation's `args_json` verified as list, `kwargs_json` verified as dict.

### Thread Safety
- `get_odoo_client()` uses double-checked locking singleton — one `OdooClient` + `requests.Session` per process
- `_DOC_CACHE` and `RUNTIME_MODEL_ISSUES` protected by `threading.Lock` for concurrent HTTP transport
- `_DOC_CACHE` limited to 100 entries with LRU eviction

### Error Sanitization
- Odoo server tracebacks (`debug` field) logged to stderr only — never included in MCP responses
- Prevents leaking file paths, SQL fragments, ORM internals to AI clients

### Docker
- Container runs as non-root user `mcp` (UID 1001)
- `run-docker.sh` uses `--env-file` instead of inline `-e` flags (prevents secret leakage in `ps`, shell history)
- `docker-compose.yml` requires `MCP_API_KEY` with `${MCP_API_KEY:?required}` syntax

### HTTP Transport
- Server **refuses to start** in streamable-http mode without `MCP_API_KEY` (`sys.exit(1)`)
- Setup wizard auto-generates cryptographically strong key (`secrets.token_urlsafe(32)`)

### Resource Limits
- `MCP_DEFAULT_CONTEXT`: 4KB max, rejected with warning if exceeded
- `MCP_BOOTSTRAP_MODELS`: 20 model cap
- `_DOC_CACHE`: 100 entry max with LRU eviction

### Files Protected by .gitignore
- `.env`, `.env.local` — environment variables
- `.mcp.json` — MCP client config (may contain API keys)
- `odoo_config.json` — legacy JSON config (may contain credentials)

## Notes for Claude Code

- This is a v2-only server - no v1 fallback code
- arg_mapping.py is essential for converting positional to named args
- Module knowledge is loaded from JSON file at startup
- Docker image must include module_knowledge.json as package data
- `@api.private` methods are blocked before API call with actionable hints (e.g. `check_access` → `has_access`, `search_fetch` → `search_read`)
- `@api.private` detection is also dynamic: methods starting with `_` are checked against live `/doc-bearer/` data when available
- `read_group` is deprecated in v19; `formatted_read_group` is the replacement (different param: `aggregates` instead of `fields`)
- `odoo://methods/{model}` is enriched with live data from `/doc-bearer/<model>.json` (signatures, return types, API decorators, exceptions, module attribution). Falls back to static data silently if unavailable.
- Live doc data requires the `api_doc.group_allow_doc` group on the API user. The `api_doc` module is auto-installed (depends: `web`).
- Live doc responses are cached in-memory for 5 minutes (`_DOC_CACHE_TTL`), max 100 entries

## Odoo MCP Best Practices

### Golden Rule: Schema First, Query Second
**NEVER guess field names. Introspect THEN query.**

**BEFORE any Odoo MCP operation on unfamiliar models:**
1. **MANDATORY**: Read `odoo://model/{model}/quick-schema` to get exact field names and types (~1.5KB, ultra-compact)
2. Read `odoo://methods/{model}` to discover available methods
3. Check `odoo://model/{model}/workflow` for state machine transitions if relevant
4. Execute with correct field names from schema

**For multiple models:** Use `odoo://bundle/model1,model2,...` (max 10) or `odoo://session-bootstrap` for common models.

**Why?** Guessing field names based on "common patterns" wastes API calls. Schema introspection is fast and gives you exact field names.

### MCP Resources Available (27 total)
| Resource | Description |
|----------|-------------|
| `odoo://models` | List all models |
| `odoo://model/{name}` | Model info with fields |
| `odoo://model/{name}/schema` | Fields and relationships |
| `odoo://model/{name}/fields` | Lightweight field list: names, types, labels |
| `odoo://model/{name}/quick-schema` | Ultra-compact schema (~1.5KB, short keys, no labels) |
| `odoo://model/{name}/workflow` | State machine transitions, methods, side effects |
| `odoo://model/{name}/docs` | Rich docs: labels, help text, selections |
| `odoo://bundle/{models}` | Batch quick-schema for N models (comma-separated, max 10) |
| `odoo://session-bootstrap` | Bootstrap conversation: schemas + workflows for common models |
| `odoo://record/{model}/{id}` | Get a specific record by ID |
| `odoo://methods/{model}` | Available methods (live-enriched with signatures, return types, API decorators from /doc-bearer/) |
| `odoo://docs/{model}` | Documentation URLs |
| `odoo://concepts` | Business term → model mappings |
| `odoo://find-model/{concept}` | Natural language → model name |
| `odoo://tools/{query}` | Search available operations by keyword |
| `odoo://actions/{model}` | Discover model actions |
| `odoo://templates` | List all resource templates (for clients without templates/list) |
| `odoo://tool-registry` | Pre-built workflows (Code-First) |
| `odoo://module-knowledge` | Special methods knowledge |
| `odoo://module-knowledge/{name}` | Knowledge for a specific module |
| `odoo://workflows` | Business workflows |
| `odoo://server/info` | Odoo server information |
| `odoo://domain-syntax` | Complete domain operator reference |
| `odoo://pagination` | Pagination guide (offset/limit/count) |
| `odoo://hierarchical` | Parent/child tree query patterns |
| `odoo://aggregation` | Aggregation guide: formatted_read_group (v19+) and read_group (deprecated) |
| `odoo://model-limitations` | Known model issues + runtime-detected problems |

### Common Errors and Fixes
| Error | Cause | Fix |
|-------|-------|-----|
| ValidationError on Many2one | Passed name not ID | Use numeric ID |
| MissingError | Record doesn't exist | Verify ID first |
| Unknown method | Typo or wrong method | get_model_methods first |
| AccessError | Permission denied | Check user permissions |
| @api.private method | Method not callable via RPC | Use public alternative (e.g. `has_access` instead of `check_access`, `search_read` instead of `search_fetch`) |

### Pre-Execution Checklist
- [ ] Model identified?
- [ ] **Fields introspected?** (odoo://model/{model}/quick-schema for compact, /fields for labels, /schema for full) ← DO THIS FIRST
- [ ] Field names from schema (not guessed)?
- [ ] Method verified (odoo://methods/{model})?
- [ ] Types correct (Many2one = ID)? Use `resolve_json` to auto-resolve names
- [ ] Required fields present? (check `required_fields` in quick-schema)
- [ ] Workflow checked? (odoo://model/{model}/workflow for state transitions)

### Key API Patterns
```python
# search_read
execute_method("model", "search_read",
    kwargs_json='{"domain": [("field", "=", "value")], "fields": ["name"], "limit": 100}')

# create
execute_method("model", "create",
    args_json='[{"field1": "value1"}]')

# write
execute_method("model", "write",
    args_json='[[ids], {"field": "new_value"}]')

# One2many commands
(0, 0, {values})  # Create new linked record
(1, id, {values}) # Update existing record
(2, id, 0)        # Delete record
(4, id, 0)        # Link existing (M2M)
(6, 0, [ids])     # Replace all (M2M)
```

## Architecture (5 Tools)

All discovery moved to resources. Action tools + resource bridge for clients without MCP resources support:

### Tools (5 total)

| Tool | Purpose |
|------|---------|
| `execute_method` | Universal Odoo API access |
| `batch_execute` | Multiple operations atomically (with progress tracking) |
| `execute_workflow` | Pre-built multi-step workflows (with progress tracking) |
| `configure_odoo` | Interactive connection configuration (user elicitation) |
| `read_resource` | Read any `odoo://` resource by URI (bridge for clients like Claude Desktop) |

### Discovery Resources

| Resource | Purpose |
|----------|---------|
| `odoo://model/{model}/quick-schema` | Ultra-compact field list (~1.5KB, short keys) — **best for tokens** |
| `odoo://model/{model}/fields` | Lightweight field list (~5-10KB, includes labels) |
| `odoo://model/{model}/schema` | Full schema with relationships (~300KB) |
| `odoo://model/{model}/workflow` | State machine transitions with side effects |
| `odoo://bundle/{models}` | Batch quick-schema for N models (max 10) |
| `odoo://session-bootstrap` | Bootstrap: schemas + workflows for common models |
| `odoo://methods/{model}` | Live-enriched methods with signatures, types, decorators (via /doc-bearer/) |
| `odoo://find-model/{concept}` | Natural language → model name |
| `odoo://tools/{query}` | Search available operations |
| `odoo://actions/{model}` | Discover model actions |
| `odoo://domain-syntax` | Domain filter reference |
| `odoo://aggregation` | Aggregation guide (formatted_read_group + read_group) |
| `odoo://pagination` | Offset/limit patterns |
| `odoo://hierarchical` | Parent/child queries |

### Workflow Examples

```python
# Multi-step workflow in one call
execute_workflow("quote_to_cash", '{"order_id": 123}')
execute_workflow("lead_to_won", '{"lead_id": 456}')
execute_workflow("create_and_post_invoice", '{"partner_id": 1, "lines": [...]}')
```

### Aggregation via execute_method

```python
# v19+ recommended: formatted_read_group (see odoo://aggregation)
execute_method("sale.order", "formatted_read_group",
  kwargs_json='{"domain": [], "groupby": ["partner_id"], "aggregates": ["amount_total:sum"]}')

# Invoice count by state
execute_method("account.move", "formatted_read_group",
  kwargs_json='{"domain": [["move_type", "=", "out_invoice"]], "groupby": ["state"], "aggregates": ["__count"]}')

# Legacy read_group (deprecated but still works)
execute_method("sale.order", "read_group",
  args_json='[[]]',
  kwargs_json='{"fields": ["amount_total:sum"], "groupby": ["partner_id"]}')
```

### Access Check

```python
# v19+ recommended: has_access (returns boolean, never raises)
execute_method("res.partner", "has_access", args_json='["read"]')

# Legacy: check_access_rights (still works)
execute_method("res.partner", "check_access_rights", args_json='["read"]')

# WARNING: check_access (without _rights) is @api.private - NOT callable via RPC
```

### MCP Prompts Available (13 total)

| Prompt | Purpose |
|--------|---------|
| `odoo-exploration` | Discover instance capabilities |
| `search-records` | Search for records in a model |
| `odoo-api-reference` | Quick API reference card |
| `quote-to-cash` | Complete sales workflow |
| `ar-aging-report` | Accounts receivable aging |
| `inventory-check` | Stock levels analysis |
| `crm-pipeline` | Pipeline analysis |
| `customer-360` | Complete customer view |
| `daily-operations` | Operations dashboard |
| `domain-builder` | Build complex domain filters |
| `hierarchical-query` | Query parent/child trees |
| `paginated-search` | Paginate large result sets |
| `aggregation-report` | Create aggregation reports (formatted_read_group / read_group) |

## Domain Operators Quick Reference

**Read `odoo://domain-syntax` for complete reference.**

| Operator | Purpose | Example |
|----------|---------|---------|
| `=`, `!=` | Equality | `["state", "=", "draft"]` |
| `>`, `<`, `>=`, `<=` | Comparison | `["amount", ">", 1000]` |
| `in`, `not in` | List | `["state", "in", ["draft", "sent"]]` |
| `like`, `ilike` | Pattern (auto %) | `["email", "ilike", "@gmail"]` |
| `=like`, `=ilike` | Exact pattern | `["code", "=like", "SO%"]` |
| `child_of` | Hierarchical children | `["category_id", "child_of", 5]` |
| `parent_of` | Hierarchical parents | `["id", "parent_of", 10]` |
| `any` | x2many match (v17+) | `["order_line", "any", [["product_id", "=", 1]]]` |

**Logic (Polish notation):**
- `["&", term1, term2]` - AND
- `["|", term1, term2]` - OR
- `["!", term]` - NOT

**Dot notation:** `["partner_id.country_id.code", "=", "US"]`

## Hierarchical Queries

**Read `odoo://hierarchical` for complete patterns.**

```python
# All descendants (uses _parent_store if available)
execute_method("product.category", "search_read",
    kwargs_json='{"domain": [["id", "child_of", 5]], "fields": ["name", "parent_id"]}')

# All ancestors
execute_method("product.category", "search_read",
    kwargs_json='{"domain": [["id", "parent_of", 10]], "fields": ["name"]}')

# Direct children only
execute_method("hr.department", "search_read",
    kwargs_json='{"domain": [["parent_id", "=", 5]], "fields": ["name"]}')

# Root records (no parent)
execute_method("product.category", "search_read",
    kwargs_json='{"domain": [["parent_id", "=", false]], "fields": ["name"]}')
```

**Hierarchical models:** product.category, account.account, hr.department, stock.location, knowledge.article

## Pagination Pattern

**Read `odoo://pagination` for complete guide.**

```python
# Step 1: Get total count
total = execute_method("res.partner", "search_count",
    kwargs_json='{"domain": [["is_company", "=", true]]}')

# Step 2: Fetch page (page 2 of 50)
execute_method("res.partner", "search_read",
    kwargs_json='{"domain": [["is_company", "=", true]], "fields": ["name"], "limit": 50, "offset": 50, "order": "name asc"}')
```

**Limits:** Default=100, Max=1000
