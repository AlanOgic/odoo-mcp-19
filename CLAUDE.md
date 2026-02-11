# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**Odoo MCP Server 19** - A standalone MCP server for Odoo 19+ using the v2 JSON-2 API.

- **Version**: 1.10.0
- **MCP Spec**: MCP 2025-11-25 (FastMCP 3.0.0b1+)
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
│   ├── __main__.py              # CLI entry point
│   ├── server.py                # MCP server (tools, resources, prompts)
│   ├── safety.py                # Safety classification engine (v1.10.0)
│   ├── odoo_client.py           # Odoo v2 API client
│   ├── arg_mapping.py           # Positional to named args conversion
│   ├── module_knowledge.json    # Module-specific methods knowledge base
│   └── assets/
│       └── odoo_icon.svg        # Odoo brand icon for MCP clients
├── Dockerfile                   # Docker build
├── docker-compose.yml           # Docker compose config
├── run-docker.sh                # Docker wrapper for Claude Desktop
├── pyproject.toml               # Package configuration
└── README.md                    # User documentation
```

### Key Components

**1. MCP Server** (`server.py`)
- **5 tools**: execute_method, batch_execute, execute_workflow, configure_odoo, read_resource
- **23 resources** for discovery (models, schema, fields, methods, actions, tools, domain-syntax, model-limitations, templates, etc.)
- **13 prompts** for guided workflows
- Module knowledge loading and error suggestions
- **Automatic fallback**: search_read → search+read on 500 errors with error categorization
- **Runtime issue tracking**: Detects and tracks problematic model/method combinations
- **Background tasks**: batch_execute and execute_workflow support progress tracking
- **Structured outputs**: All tools return typed Pydantic models with execution time
- Smart limits: DEFAULT_LIMIT=100, MAX_LIMIT=1000

**2. Odoo Client** (`odoo_client.py`)
- v2 JSON-2 API only (Bearer token auth)
- Endpoint: `/json/2/{model}/{method}`
- Automatic argument conversion via arg_mapping
- Live model documentation via `/doc-bearer/<model>.json` (api_doc module)

**3. Argument Mapping** (`arg_mapping.py`)
- Converts positional args to named args for v2 API
- Supports 30 ORM methods (search, create, write, formatted_read_group, has_access, action_*, button_*, etc.)

**4. Module Knowledge** (`module_knowledge.json`)
- Special methods for 13 Odoo modules (including AI module)
- 30 ORM methods documented
- Error patterns with suggestions
- Validated against Odoo 19 source code

**5. AI Module** (Enterprise only)
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
- `MCP_API_KEY` - Bearer token for HTTP authentication
- `MCP_HOST` - HTTP bind address (default: 0.0.0.0)
- `MCP_PORT` - HTTP port (default: 8080)

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

### Confirmation Flow

1. Caller sends `execute_method(model, method, args_json, kwargs_json)`
2. Safety layer classifies the operation
3. If confirmation needed → returns `pending_confirmation=true` with `safety` classification
4. Caller reviews, then re-calls with `confirmed=true` to proceed

### Blocked Models (write always refused)

`ir.rule`, `ir.model.access`, `ir.module.module`, `ir.config_parameter`, `res.users`

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

Safety checks are integrated in `execute_method`, `batch_execute`, and `execute_workflow`. The `confirmed` parameter (default `False`) is backward-compatible.

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
- Live doc responses are cached in-memory for 5 minutes (`_DOC_CACHE_TTL`)

## Odoo MCP Best Practices

### Golden Rule: Schema First, Query Second
**NEVER guess field names. Introspect THEN query.**

**BEFORE any Odoo MCP operation on unfamiliar models:**
1. **MANDATORY**: Read `odoo://model/{model}/schema` to get exact field names and types
2. Read `odoo://methods/{model}` to discover available methods
3. Check `odoo://docs/{model}` resource for documentation URLs if needed
4. Execute with correct field names from schema

**Why?** Guessing field names based on "common patterns" wastes API calls. Schema introspection is fast and gives you exact field names.

### MCP Resources Available (23 total)
| Resource | Description |
|----------|-------------|
| `odoo://models` | List all models |
| `odoo://model/{name}` | Model info with fields |
| `odoo://model/{name}/schema` | Fields and relationships |
| `odoo://model/{name}/fields` | Lightweight field list: names, types, labels |
| `odoo://model/{name}/docs` | Rich docs: labels, help text, selections |
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
- [ ] **Fields introspected?** (odoo://model/{model}/fields for lightweight, or /schema for full) ← DO THIS FIRST
- [ ] Field names from schema (not guessed)?
- [ ] Method verified (odoo://methods/{model})?
- [ ] Types correct (Many2one = ID)?
- [ ] Required fields present?

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
| `odoo://model/{model}/fields` | Lightweight field list (~5-10KB) |
| `odoo://model/{model}/schema` | Full schema with relationships (~300KB) |
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
