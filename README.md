# Odoo MCP Server for Odoo 19+

A Model Context Protocol (MCP) server for interacting with Odoo 19+ via the JSON-2 API.

## Features

- **v2 JSON-2 API only** - Optimized for Odoo 19+
- **3 universal tools** - `execute_method`, `batch_execute`, and `get_model_methods`
- **Discovery resources** - Models, schemas, workflows, module knowledge
- **Smart limits** - Automatic pagination to prevent massive data returns
- **Module knowledge base** - Special methods for 10+ Odoo modules (sale, account, crm, stock, etc.)
- **Smart error suggestions** - Contextual help for common errors

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

## Resources

| Resource | Description |
|----------|-------------|
| `odoo://models` | List all models |
| `odoo://model/{name}` | Model info with fields |
| `odoo://model/{name}/schema` | Complete schema |
| `odoo://record/{model}/{id}` | Get specific record |
| `odoo://methods/{model}` | Available methods |
| `odoo://workflows` | Business workflows |
| `odoo://server/info` | Server information |
| `odoo://module-knowledge` | All module-specific knowledge |
| `odoo://module-knowledge/{name}` | Knowledge for specific module |

## Module Knowledge

The server includes built-in knowledge for 10 Odoo modules with their special methods:

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
