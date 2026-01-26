# Odoo MCP Server for Odoo 19+

A Model Context Protocol (MCP) server for interacting with Odoo 19+ via the JSON-2 API.

## Features

- **v2 JSON-2 API only** - Optimized for Odoo 19+
- **8 powerful tools** - Core tools + Code-First Pattern for 98% token reduction
- **12+ discovery resources** - Models, schemas, workflows, docs, concepts
- **Smart limits** - Automatic pagination to prevent massive data returns
- **Module knowledge base** - Special methods for 12+ Odoo modules
- **Smart error suggestions** - Contextual help for common errors
- **Natural language model discovery** - "invoice" → `account.move`
- **Multi-step workflows** - Execute complex operations in single calls
- **Aggregation support** - Efficient `read_group` for reporting

## Quick Start

### 1. Install

```bash
pip install odoo-mcp-19
```

Or install from source:

```bash
cd /path/to/odoo-mcp-19
pip install -e .
```

### 2. Configure

Create a `.env` file:

```env
ODOO_URL=https://your-instance.odoo.com
ODOO_DB=your-database
ODOO_USERNAME=your-username
ODOO_API_KEY=your-api-key
```

Or use environment variables directly.

### 3. Add to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "odoo": {
      "command": "odoo-mcp-19",
      "env": {
        "ODOO_URL": "https://your-instance.odoo.com",
        "ODOO_DB": "your-database",
        "ODOO_USERNAME": "your-username",
        "ODOO_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Tools

### `execute_method`

Execute any Odoo method on any model:

```
model: res.partner
method: search_read
args_json: [[["is_company", "=", true]]]
kwargs_json: {"fields": ["name", "email"], "limit": 10}
```

### `batch_execute`

Execute multiple operations atomically:

```json
{
  "operations": [
    {"model": "res.partner", "method": "create", "args_json": "[{\"name\": \"Test\"}]"},
    {"model": "res.partner", "method": "search_read", "args_json": "[[]]", "kwargs_json": "{\"limit\": 5}"}
  ],
  "atomic": true
}
```

### `get_model_methods`

Discover available methods on a model, including special module-specific methods:

```
model: sale.order
```

Returns standard ORM methods plus special methods like `action_confirm`, `_create_invoices`, etc.

### `find_model`

Find Odoo model from natural language:

```
concept: invoice  →  account.move
concept: customer →  res.partner
concept: quote    →  sale.order
```

### `search_tools` (Code-First Pattern)

Search available operations by keyword - loads tools on-demand for 98% token reduction:

```
query: invoice
→ Returns: create_invoice, post_invoice, get_overdue_invoices, get_ar_aging
```

### `discover_model_actions`

Discover ALL available actions for a model from registry, knowledge base, and server actions:

```
model: sale.order
→ Returns: workflows, special_methods, server_actions, usage_examples
```

### `execute_workflow`

Execute multi-step workflows in a single call:

```
workflow: quote_to_cash
params_json: {"order_id": 123}
→ Executes: Confirm order → Create invoice → Post invoice
```

Available workflows: `quote_to_cash`, `lead_to_won`, `create_and_post_invoice`

### `aggregate_data`

Efficient database-level aggregation using `read_group`:

```
model: sale.order
groupby: partner_id
fields: amount_total:sum
→ Returns: Total sales by customer
```

## Resources

| Resource | Description |
|----------|-------------|
| `odoo://models` | List all models |
| `odoo://model/{name}` | Model info with fields |
| `odoo://model/{name}/schema` | Complete schema with relationships |
| `odoo://model/{name}/docs` | Rich docs: labels, help text, selections |
| `odoo://record/{model}/{id}` | Get specific record |
| `odoo://methods/{model}` | Available methods |
| `odoo://docs/{model}` | Documentation URLs (Odoo docs, GitHub) |
| `odoo://workflows` | Business workflows |
| `odoo://server/info` | Server information |
| `odoo://concepts` | Business term → model mappings |
| `odoo://tool-registry` | Pre-built workflows (Code-First) |
| `odoo://module-knowledge` | All module-specific knowledge |
| `odoo://module-knowledge/{name}` | Knowledge for specific module |

## Prompts

Pre-built workflow templates:

| Prompt | Description |
|--------|-------------|
| `odoo-exploration` | Discover Odoo instance capabilities |
| `search-records` | Search records in any model |
| `odoo-api-reference` | Quick API reference |
| `quote-to-cash` | Complete sales workflow |
| `ar-aging-report` | Accounts receivable aging |
| `inventory-check` | Stock levels analysis |
| `crm-pipeline` | Pipeline analysis |
| `customer-360` | Complete customer view |
| `daily-operations` | Operations dashboard |

## Module Knowledge

The server includes built-in knowledge for 12 Odoo modules with their special methods:

| Module | Model | Special Methods |
|--------|-------|-----------------|
| `knowledge` | knowledge.article | `article_create`, `article_duplicate` |
| `sale` | sale.order | `action_confirm`, `action_cancel`, `_create_invoices` |
| `account` | account.move | `action_post`, `button_draft`, `action_register_payment` |
| `crm` | crm.lead | `action_set_won`, `action_set_lost`, `convert_opportunity` |
| `stock` | stock.picking | `button_validate`, `action_confirm`, `action_assign` |
| `purchase` | purchase.order | `button_confirm`, `action_create_invoice` |
| `hr_expense` | hr.expense | `action_submit_expenses`, `action_approve_expense_sheets` |
| `hr_leave` | hr.leave | `action_approve`, `action_refuse` |
| `project` | project.task | `action_assign_to_me` |
| `documents` | documents.document | `document_create` |
| `discuss` | discuss.channel | `channel_create`, `add_members` |
| `mail` | mail.message | Direct create (message_type required) |

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `ODOO_URL` | Odoo server URL | Required |
| `ODOO_DB` | Database name | Required |
| `ODOO_USERNAME` | Username | Required |
| `ODOO_API_KEY` | API key (recommended) | - |
| `ODOO_PASSWORD` | Password (fallback) | - |
| `ODOO_TIMEOUT` | Request timeout (seconds) | 30 |
| `ODOO_VERIFY_SSL` | Verify SSL certificates | true |

## Docker

### Build

```bash
docker build -t odoo-mcp-19 .
```

### Use with Claude Desktop (Docker)

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "odoo19": {
      "command": "/path/to/odoo-mcp-19/run-docker.sh",
      "env": {
        "ODOO_URL": "https://your-instance.odoo.com",
        "ODOO_DB": "your-database",
        "ODOO_USERNAME": "your-username",
        "ODOO_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Test manually

```bash
# With .env file
cp .env.example .env
# Edit .env with your credentials
docker compose run --rm odoo-mcp

# Or with inline env vars
docker run --rm -i \
  -e ODOO_URL=https://your-instance.odoo.com \
  -e ODOO_DB=your-db \
  -e ODOO_USERNAME=your-user \
  -e ODOO_API_KEY=your-key \
  odoo-mcp-19
```

## Requirements

- Python 3.10+ (or Docker)
- Odoo 19+
- FastMCP 2.13.0+

## License

MIT
