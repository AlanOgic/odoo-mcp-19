# Odoo MCP Server for Odoo 19+

Connect Claude and AI assistants to Odoo 19+ via the Model Context Protocol (MCP).

## Features

- **3 tools** - `execute_method`, `batch_execute`, `execute_workflow`
- **17+ resources** - Dynamic model discovery and introspection
- **30 ORM methods** - Complete documentation with examples
- **13 modules** - Special methods including AI module (Enterprise)

## Quick start

### 1. Install

```bash
pip install odoo-mcp-19
```

### 2. Configure Claude Desktop

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

### 3. Use it

Ask Claude: *"List the first 5 partners in Odoo"*

## Example

```python
# Search and read partners
execute_method(
    model="res.partner",
    method="search_read",
    kwargs_json='{"domain": [["is_company", "=", true]], "fields": ["name", "email"], "limit": 10}'
)
```

## Documentation

Full documentation available in the **[Wiki](https://github.com/AlanOgic/odoo-mcp-19/wiki)**:

- [Getting Started](https://github.com/AlanOgic/odoo-mcp-19/wiki/Getting-Started) - Installation and configuration
- [Tools](https://github.com/AlanOgic/odoo-mcp-19/wiki/Tools) - The 3 available tools
- [Resources](https://github.com/AlanOgic/odoo-mcp-19/wiki/Resources) - Discovery and introspection
- [ORM Methods](https://github.com/AlanOgic/odoo-mcp-19/wiki/ORM-Methods) - 30 documented methods
- [Module Knowledge](https://github.com/AlanOgic/odoo-mcp-19/wiki/Module-Knowledge) - Special methods
- [AI Module](https://github.com/AlanOgic/odoo-mcp-19/wiki/AI-Module) - Odoo 19 AI (Enterprise)
- [Domain Syntax](https://github.com/AlanOgic/odoo-mcp-19/wiki/Domain-Syntax) - Search filters
- [Prompts](https://github.com/AlanOgic/odoo-mcp-19/wiki/Prompts) - Guided workflows

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `ODOO_URL` | Yes | Odoo server URL |
| `ODOO_DB` | Yes | Database name |
| `ODOO_USERNAME` | Yes | Username |
| `ODOO_API_KEY` | Yes | API key |
| `ODOO_TIMEOUT` | No | Timeout seconds (default: 30) |

## Requirements

- Python 3.10+
- Odoo 19+
- FastMCP 3.0.0+

## License

MIT
