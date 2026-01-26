# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**Odoo MCP Server 19** - A standalone MCP server for Odoo 19+ using the v2 JSON-2 API.

- **Version**: 1.0.0
- **MCP Spec**: MCP 0.2 (FastMCP 2.13.0+)
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
- 8 tools: Core + Code-First Pattern + Aggregation
- 12+ resources for discovery
- Module knowledge loading and error suggestions
- Smart limits: DEFAULT_LIMIT=100, MAX_LIMIT=1000
- Code-First Pattern for 98% token reduction

**2. Odoo Client** (`odoo_client.py`)
- v2 JSON-2 API only (Bearer token auth)
- Endpoint: `/json/2/{model}/{method}`
- Automatic argument conversion via arg_mapping

**3. Argument Mapping** (`arg_mapping.py`)
- Converts positional args to named args for v2 API
- Supports 17+ ORM methods (search, create, write, etc.)

**4. Module Knowledge** (`module_knowledge.json`)
- Special methods for 10 Odoo modules
- Error patterns with suggestions

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

### Golden Rule
**BEFORE any Odoo MCP operation on unfamiliar models:**
1. Call `get_model_methods(model)` to discover available methods
2. Check `odoo://docs/{model}` resource for documentation URLs if needed
3. Execute with correct parameters

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

### Common Errors and Fixes
| Error | Cause | Fix |
|-------|-------|-----|
| ValidationError on Many2one | Passed name not ID | Use numeric ID |
| MissingError | Record doesn't exist | Verify ID first |
| Unknown method | Typo or wrong method | get_model_methods first |
| AccessError | Permission denied | Check user permissions |

### Pre-Execution Checklist
- [ ] Model identified?
- [ ] Method verified (get_model_methods)?
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

## Code-First Pattern (98% Token Reduction)

Instead of loading all tool definitions upfront, use on-demand discovery:

### New High-Impact Tools

| Tool | Purpose | Impact |
|------|---------|--------|
| `search_tools(query)` | Find operations by keyword | Very High |
| `discover_model_actions(model)` | All actions for a model | Very High |
| `execute_workflow(name, params)` | Multi-step workflows | Very High |
| `aggregate_data(model, groupby, fields)` | Efficient reporting | High |
| `find_model(concept)` | Natural language → model | High |

### Usage Pattern

```python
# OLD: Call many individual methods
execute_method("sale.order", "action_confirm", ...)
execute_method("sale.order", "_create_invoices", ...)
execute_method("account.move", "action_post", ...)

# NEW: Single workflow call
execute_workflow("quote_to_cash", '{"order_id": 123}')
```

### Available Workflows

| Workflow | Description |
|----------|-------------|
| `quote_to_cash` | Confirm order → Create invoice → Post |
| `lead_to_won` | Convert lead → Mark opportunity won |
| `create_and_post_invoice` | Create invoice with lines → Post |

### Aggregation Examples

```python
# Total sales by customer
aggregate_data("sale.order", "partner_id", "amount_total:sum")

# Invoice count by state
aggregate_data("account.move", "state", "__count")

# Revenue by month
aggregate_data("sale.order", "date_order:month", "amount_total:sum")
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
