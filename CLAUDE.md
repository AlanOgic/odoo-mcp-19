# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**Odoo MCP Server 19** - A standalone MCP server for Odoo 19+ using the v2 JSON-2 API.

- **Version**: 1.6.0
- **MCP Spec**: MCP 0.2 (FastMCP 3.0.0+)
- **Odoo Support**: v19+ only (v2 JSON-2 API)

## Architecture

### Package Structure

```
odoo-mcp-19/
├── src/odoo_mcp/
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # CLI entry point
│   ├── server.py                # MCP server (tools, resources, prompts)
│   ├── odoo_client.py           # Odoo v2 API client
│   ├── arg_mapping.py           # Positional to named args conversion
│   └── module_knowledge.json    # Module-specific methods knowledge base
├── Dockerfile                   # Docker build
├── docker-compose.yml           # Docker compose config
├── run-docker.sh                # Docker wrapper for Claude Desktop
├── pyproject.toml               # Package configuration
└── README.md                    # User documentation
```

### Key Components

**1. MCP Server** (`server.py`)
- **3 tools only**: execute_method, batch_execute, execute_workflow
- **17 resources** for discovery (models, schema, methods, actions, tools, domain-syntax, etc.)
- **14 prompts** for guided workflows
- Module knowledge loading and error suggestions
- Smart limits: DEFAULT_LIMIT=100, MAX_LIMIT=1000

**2. Odoo Client** (`odoo_client.py`)
- v2 JSON-2 API only (Bearer token auth)
- Endpoint: `/json/2/{model}/{method}`
- Automatic argument conversion via arg_mapping

**3. Argument Mapping** (`arg_mapping.py`)
- Converts positional args to named args for v2 API
- Supports 17+ ORM methods (search, create, write, etc.)

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

## Notes for Claude Code

- This is a v2-only server - no v1 fallback code
- arg_mapping.py is essential for converting positional to named args
- Module knowledge is loaded from JSON file at startup
- Docker image must include module_knowledge.json as package data

## Odoo MCP Best Practices

### Golden Rule: Schema First, Query Second
**NEVER guess field names. Introspect THEN query.**

**BEFORE any Odoo MCP operation on unfamiliar models:**
1. **MANDATORY**: Read `odoo://model/{model}/schema` to get exact field names and types
2. Read `odoo://methods/{model}` to discover available methods
3. Check `odoo://docs/{model}` resource for documentation URLs if needed
4. Execute with correct field names from schema

**Why?** Guessing field names based on "common patterns" wastes API calls. Schema introspection is fast and gives you exact field names.

### MCP Resources Available
| Resource | Description |
|----------|-------------|
| `odoo://models` | List all models |
| `odoo://model/{name}/schema` | Fields and relationships |
| `odoo://model/{name}/docs` | Rich docs: labels, help text, selections |
| `odoo://methods/{model}` | Available methods |
| `odoo://docs/{model}` | Documentation URLs |
| `odoo://module-knowledge` | Special methods knowledge |
| `odoo://workflows` | Business workflows |
| `odoo://concepts` | Business term → model mappings |
| `odoo://tool-registry` | Pre-built workflows (Code-First) |
| `odoo://domain-syntax` | **Complete domain operator reference** |
| `odoo://pagination` | **Pagination guide (offset/limit/count)** |
| `odoo://hierarchical` | **Parent/child tree query patterns** |
| `odoo://aggregation` | **read_group aggregation reference** |

### Common Errors and Fixes
| Error | Cause | Fix |
|-------|-------|-----|
| ValidationError on Many2one | Passed name not ID | Use numeric ID |
| MissingError | Record doesn't exist | Verify ID first |
| Unknown method | Typo or wrong method | get_model_methods first |
| AccessError | Permission denied | Check user permissions |

### Pre-Execution Checklist
- [ ] Model identified?
- [ ] **Schema introspected?** (odoo://model/{model}/schema) ← DO THIS FIRST
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

## Simplified Architecture (Only 3 Tools)

All discovery moved to resources. Only action tools remain:

### Tools (3 total)

| Tool | Purpose |
|------|---------|
| `execute_method` | Universal Odoo API access |
| `batch_execute` | Multiple operations atomically |
| `execute_workflow` | Pre-built multi-step workflows |

### Discovery Resources

| Resource | Purpose |
|----------|---------|
| `odoo://find-model/{concept}` | Natural language → model name |
| `odoo://tools/{query}` | Search available operations |
| `odoo://actions/{model}` | Discover model actions |
| `odoo://domain-syntax` | Domain filter reference |
| `odoo://aggregation` | read_group guide |
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
# Total sales by customer (see odoo://aggregation)
execute_method("sale.order", "read_group",
  args_json='[[]]',
  kwargs_json='{"fields": ["amount_total:sum"], "groupby": ["partner_id"]}')

# Invoice count by state
execute_method("account.move", "read_group",
  args_json='[[["move_type", "=", "out_invoice"]]]',
  kwargs_json='{"fields": ["__count"], "groupby": ["state"]}')
```

### MCP Prompts Available

| Prompt | Purpose |
|--------|---------|
| `quote-to-cash` | Complete sales workflow |
| `ar-aging-report` | Accounts receivable aging |
| `inventory-check` | Stock levels analysis |
| `crm-pipeline` | Pipeline analysis |
| `customer-360` | Complete customer view |
| `daily-operations` | Operations dashboard |
| `domain-builder` | **Build complex domain filters** |
| `hierarchical-query` | **Query parent/child trees** |
| `paginated-search` | **Paginate large result sets** |
| `aggregation-report` | **Create read_group reports** |

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
