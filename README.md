# Odoo MCP Server for Odoo 19+

A Model Context Protocol (MCP) server for interacting with Odoo 19+ via the JSON-2 API.

## Features

- **v2 JSON-2 API only** - Optimized for Odoo 19+ (XML-RPC deprecated in Odoo 20)
- **Only 3 tools** - Minimal interface: `execute_method`, `batch_execute`, `execute_workflow`
- **17+ discovery resources** - Models, schemas, methods, docs, workflows, domain syntax
- **14 guided prompts** - Business workflows, reporting, domain building
- **Comprehensive ORM documentation** - All methods with parameters and examples
- **Module knowledge base** - Special methods for 12+ Odoo modules
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

## Tools (Only 3)

### `execute_method`

Execute any Odoo method on any model:

```python
# Search and read partners
execute_method(
    model="res.partner",
    method="search_read",
    kwargs_json='{"domain": [["is_company", "=", true]], "fields": ["name", "email"], "limit": 10}'
)

# Create a record
execute_method(
    model="res.partner",
    method="create",
    args_json='[{"name": "New Partner", "email": "test@example.com"}]'
)

# Aggregation with read_group
execute_method(
    model="sale.order",
    method="read_group",
    args_json='[[]]',
    kwargs_json='{"fields": ["amount_total:sum"], "groupby": ["partner_id"], "lazy": false}'
)
```

### `batch_execute`

Execute multiple operations atomically:

```json
{
  "operations": [
    {"model": "res.partner", "method": "create", "args_json": "[{\"name\": \"Test\"}]"},
    {"model": "res.partner", "method": "search_read", "kwargs_json": "{\"limit\": 5}"}
  ],
  "atomic": true
}
```

### `execute_workflow`

Execute multi-step workflows in a single call:

```python
# Quote to cash workflow
execute_workflow("quote_to_cash", '{"order_id": 123}')
# Executes: Confirm order → Create invoice → Post invoice

# Lead to won
execute_workflow("lead_to_won", '{"lead_id": 456}')
```

Available workflows: `quote_to_cash`, `lead_to_won`, `create_and_post_invoice`, `stock_transfer`

## Resources (Discovery)

| Resource | Description |
|----------|-------------|
| `odoo://models` | List all models |
| `odoo://model/{name}/schema` | Complete schema with field types |
| `odoo://model/{name}/docs` | Rich docs: labels, help text, selections |
| `odoo://methods/{model}` | Available methods (read, write, introspection, special) |
| `odoo://find-model/{concept}` | Natural language → model ("invoice" → account.move) |
| `odoo://actions/{model}` | Discover all actions for a model |
| `odoo://tools/{query}` | Search available operations |
| `odoo://domain-syntax` | Complete domain operator reference |
| `odoo://aggregation` | read_group guide with lazy parameter |
| `odoo://pagination` | Offset/limit patterns |
| `odoo://hierarchical` | Parent/child tree queries |
| `odoo://workflows` | Business workflows |
| `odoo://concepts` | Business term → model mappings |
| `odoo://module-knowledge` | All module-specific knowledge |

## Prompts

| Prompt | Description |
|--------|-------------|
| `odoo-exploration` | Discover Odoo instance capabilities |
| `odoo-api-reference` | Quick API reference |
| `quote-to-cash` | Complete sales workflow |
| `ar-aging-report` | Accounts receivable aging |
| `inventory-check` | Stock levels analysis |
| `crm-pipeline` | Pipeline analysis |
| `customer-360` | Complete customer view |
| `domain-builder` | Build complex domain filters |
| `hierarchical-query` | Query parent/child trees |
| `paginated-search` | Pagination patterns |
| `aggregation-report` | read_group reporting |

## ORM Methods Documented

The module knowledge includes comprehensive documentation for:

| Method | Description |
|--------|-------------|
| `search` | Search for record IDs with domain filter |
| `search_count` | Count matching records (lightweight) |
| `search_read` | Search and read in one call (most efficient) |
| `read` | Read specific records by ID |
| `create` | Create new record(s) - supports batch |
| `write` | Update existing record(s) |
| `unlink` | Delete record(s) |
| `read_group` | Aggregation with grouping |
| `name_search` | Autocomplete search for Many2one |
| `default_get` | Get default values for fields |
| `fields_get` | Get field definitions |
| `copy` | Duplicate a record |
| `check_access_rights` | Check user permissions |

## Module Knowledge

Built-in knowledge for 12+ Odoo modules with special methods:

| Module | Model | Special Methods |
|--------|-------|-----------------|
| `sale` | sale.order | `action_confirm`, `action_cancel`, `_create_invoices` |
| `account` | account.move | `action_post`, `button_draft`, `action_register_payment` |
| `crm` | crm.lead | `action_set_won`, `action_set_lost`, `convert_opportunity` |
| `stock` | stock.picking | `button_validate`, `action_confirm`, `action_assign` |
| `purchase` | purchase.order | `button_confirm`, `action_create_invoice` |
| `knowledge` | knowledge.article | `article_create`, `article_duplicate` |
| `hr_expense` | hr.expense | `action_submit_expenses` |
| `hr_leave` | hr.leave | `action_approve`, `action_refuse` |

## Configuration

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

### Use with Claude Desktop

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

## Requirements

- Python 3.10+
- Odoo 19+
- FastMCP 3.0.0+

## License

MIT
