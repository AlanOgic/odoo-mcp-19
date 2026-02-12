```
   ____      __               __  _____________     _______
  / __ \____/ /___  ____     /  |/  / ____/ __ \   <  / __ \  __
 / / / / __  / __ \/ __ \   / /|_/ / /   / /_/ /   / / /_/ /_/ /_
/ /_/ / /_/ / /_/ / /_/ /  / /  / / /___/ ____/   / /\__, /_  __/
\____/\__,_/\____/\____/  /_/  /_/\____/_/       /_//____/ /_/
```

Connect Claude and AI assistants to Odoo 19+ via the Model Context Protocol (MCP).

## Features

- **5 tools, full power** - `execute_method` calls ANY method on ANY model. Combined with `batch_execute`, `execute_workflow`, `configure_odoo`, and `read_resource`, you have complete Odoo API access
- **27 resources** - Dynamic model discovery, compact schemas, workflows, and introspection
- **13 prompts** - Guided workflows for common business operations
- **30 ORM methods** - Complete documentation with examples
- **13 modules** - Special methods including AI module (Enterprise)
- **Safety layer** - Pre-execution risk classification, blocked models, cascade warnings
- **DX optimizations** - Quick-schema, bundle, session-bootstrap, resolve_json for token-efficient AI operations
- **MCP 2025-11-25** - Background tasks, progress tracking, icons, structured outputs

## Installation

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (recommended) **OR** Python 3.10+
- An Odoo 19+ instance with API access
- An API key from your Odoo instance (Settings > Users > API Keys)

### Option A: Docker (recommended)

**Pull from Docker Hub:**

```bash
docker pull alanogik/odoo-mcp-19:latest
```

**Or build from source:**

```bash
git clone https://github.com/AlanOgic/odoo-mcp-19.git
cd odoo-mcp-19
docker build -t odoo-mcp-19:latest .
```

**Quick test (verify it connects):**

```bash
docker run --rm -i \
  -e ODOO_URL=https://your-instance.odoo.com \
  -e ODOO_DB=your-database \
  -e ODOO_USERNAME=your-username \
  -e ODOO_API_KEY=your-api-key \
  odoo-mcp-19:latest
```

You should see the server start without errors. Press `Ctrl+C` to stop.

### Option B: From source (Python venv)

```bash
# 1. Clone the repository
git clone https://github.com/AlanOgic/odoo-mcp-19.git
cd odoo-mcp-19

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 4. Install the package
pip install -e .

# 5. Create your .env file
cp .env.example .env             # Then edit with your Odoo credentials
# Or create manually:
cat > .env << 'EOF'
ODOO_URL=https://your-instance.odoo.com
ODOO_DB=your-database
ODOO_USERNAME=your-username
ODOO_API_KEY=your-api-key
EOF

# 6. Test it
python -m odoo_mcp
```

### Option C: pip install

```bash
pip install odoo-mcp-19
```

## Configure Claude Desktop

Edit your Claude Desktop configuration file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Using Docker (recommended)

**Method 1: Using `run-docker.sh` wrapper (simplest)**

Create a `.env` file in the project root with your credentials, then:

```json
{
  "mcpServers": {
    "odoo": {
      "command": "/path/to/odoo-mcp-19/run-docker.sh"
    }
  }
}
```

**Method 2: Inline Docker command**

```json
{
  "mcpServers": {
    "odoo": {
      "command": "docker",
      "args": ["run", "--rm", "-i",
        "-e", "ODOO_URL=https://your-instance.odoo.com",
        "-e", "ODOO_DB=your-database",
        "-e", "ODOO_USERNAME=your-username",
        "-e", "ODOO_API_KEY=your-api-key",
        "odoo-mcp-19:latest"
      ]
    }
  }
}
```

### Using Python (venv)

```json
{
  "mcpServers": {
    "odoo": {
      "command": "/path/to/odoo-mcp-19/.venv/bin/python",
      "args": ["-m", "odoo_mcp"],
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

### Using pip install

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

**Restart Claude Desktop** after saving the config file.

### Verify it works

Ask Claude: *"List the first 5 partners in Odoo"*

## Quick start examples

```python
# Search partners
execute_method("res.partner", "search_read",
    kwargs_json='{"domain": [["is_company", "=", true]], "fields": ["name", "email"], "limit": 10}')

# Create a partner
execute_method("res.partner", "create",
    args_json='[{"name": "ACME Corp", "is_company": true, "email": "info@acme.com"}]')

# Update with auto-resolved Many2one (no need to know user ID)
execute_method("res.partner", "write",
    args_json='[[42], {"name": "ACME Corp Updated"}]',
    resolve_json='{"user_id": {"model": "res.users", "search": "John"}}')

# Confirm a sale order (with safety confirmation)
execute_method("sale.order", "action_confirm", args_json='[[15]]', confirmed=true)

# Multi-step workflow in one call
execute_workflow("quote_to_cash", '{"order_id": 123}')
```

## Architecture

### Tools (5)

| Tool | Purpose |
|------|---------|
| `execute_method` | Call any method on any Odoo model |
| `batch_execute` | Multiple operations with progress tracking |
| `execute_workflow` | Pre-built multi-step workflows |
| `configure_odoo` | Interactive connection setup |
| `read_resource` | Read any `odoo://` resource by URI |

### Resources (27)

| Resource | Description |
|----------|-------------|
| `odoo://models` | List all models |
| `odoo://model/{name}` | Model info with fields |
| `odoo://model/{name}/schema` | Full fields and relationships |
| `odoo://model/{name}/fields` | Lightweight field list with labels |
| `odoo://model/{name}/quick-schema` | Ultra-compact schema (~1.5KB, short keys) |
| `odoo://model/{name}/workflow` | State machine transitions and side effects |
| `odoo://model/{name}/docs` | Rich docs: labels, help text, selections |
| `odoo://bundle/{models}` | Batch quick-schema for N models (max 10) |
| `odoo://session-bootstrap` | Bootstrap: schemas + workflows for common models |
| `odoo://record/{model}/{id}` | Get a specific record by ID |
| `odoo://methods/{model}` | Available methods (enriched with live signatures) |
| `odoo://docs/{model}` | Documentation URLs |
| `odoo://concepts` | Business term to model mappings |
| `odoo://find-model/{concept}` | Natural language to model name |
| `odoo://tools/{query}` | Search available operations |
| `odoo://actions/{model}` | Discover model actions |
| `odoo://templates` | List all resource templates |
| `odoo://tool-registry` | Pre-built workflows |
| `odoo://module-knowledge` | Special methods knowledge |
| `odoo://module-knowledge/{name}` | Knowledge for a specific module |
| `odoo://workflows` | Business workflows |
| `odoo://server/info` | Odoo server information |
| `odoo://domain-syntax` | Domain operator reference |
| `odoo://pagination` | Pagination guide |
| `odoo://hierarchical` | Parent/child tree query patterns |
| `odoo://aggregation` | Aggregation guide (formatted_read_group) |
| `odoo://model-limitations` | Known model issues + runtime problems |

### Prompts (13)

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
| `aggregation-report` | Aggregation reports |

## Safety Layer (v1.10.0)

Pre-execution safety classification gates dangerous operations behind confirmation.

### Risk levels

| Level | Behavior | Confirm? |
|-------|----------|----------|
| `SAFE` | Execute immediately | Never |
| `MEDIUM` | Gate based on mode/volume | Conditional |
| `HIGH` | Always require confirmation | Always |
| `BLOCKED` | Always refuse | N/A |

### Blocked models (write always refused)

`ir.rule`, `ir.model.access`, `ir.module.module`, `ir.config_parameter`, `res.users`

### Sensitive models (write always confirms)

`account.move`, `account.payment`, `account.bank.statement`, `hr.payslip`, `ir.cron`

### Cascade warnings

Side effects are surfaced for workflow actions:
- `sale.order` + `action_confirm` → creates deliveries
- `account.move` + `action_post` → creates journal entries (irreversible)
- `stock.picking` + `button_validate` → updates stock levels
- `purchase.order` + `button_confirm` → creates incoming receipts
- `account.payment` + `action_post` → creates journal entries + reconciliation

### Confirmation flow

1. Caller sends `execute_method(model, method, args_json)`
2. Safety layer classifies the operation
3. If confirmation needed: returns `pending_confirmation=true` with `safety` classification
4. Caller reviews, then re-calls with `confirmed=true` to proceed

## DX Improvements (v1.11.0)

### Quick Schema

`odoo://model/{model}/quick-schema` — Ultra-compact schema with short keys: `t` (type), `req` (required), `ro` (readonly), `rel` (relation). ~60-80% smaller than `/fields`.

### Bundle

`odoo://bundle/res.partner,sale.order,stock.picking` — Batch quick-schema for up to 10 models in one call.

### Session Bootstrap

`odoo://session-bootstrap` — One call to bootstrap a conversation with schemas + workflows for common models. Configure via `MCP_BOOTSTRAP_MODELS` env var.

### Workflow

`odoo://model/{model}/workflow` — State machine transitions for 6 main models with side effects and irreversibility flags. Dynamic fallback for unmapped models.

### Many2one Resolution

Auto-resolve field names to IDs with `resolve_json`:

```python
execute_method("res.partner", "write",
    args_json='[[1], {"user_id": null}]',
    resolve_json='{"user_id": {"model": "res.users", "search": "John"}}')
```

### Default Context

Set `MCP_DEFAULT_CONTEXT` to apply context to all operations:

```bash
export MCP_DEFAULT_CONTEXT='{"lang": "fr_FR", "tz": "Europe/Paris"}'
```

### Expanded Error Patterns

~25 error patterns with actionable suggestions covering 422, 500, 403, 404 errors and fallback patterns.

## HTTP Transport

Run as HTTP server with Bearer token authentication:

```bash
docker run -d -p 8080:8080 \
  -e ODOO_URL=https://your.odoo.com \
  -e ODOO_DB=mydb \
  -e ODOO_USERNAME=admin \
  -e ODOO_API_KEY=xxx \
  -e MCP_TRANSPORT=streamable-http \
  -e MCP_API_KEY=your-secret-token \
  odoo-mcp-19:latest
```

Connect with: `Authorization: Bearer your-secret-token`

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ODOO_URL` | Yes | — | Odoo server URL |
| `ODOO_DB` | Yes | — | Database name |
| `ODOO_USERNAME` | Yes | — | Username |
| `ODOO_API_KEY` | Yes | — | API key (Settings > Users > API Keys) |
| `ODOO_PASSWORD` | No | — | Password (fallback if no API key) |
| `ODOO_TIMEOUT` | No | `30` | Request timeout in seconds |
| `ODOO_VERIFY_SSL` | No | `true` | SSL certificate verification |
| `MCP_TRANSPORT` | No | `stdio` | Transport: `stdio` or `streamable-http` |
| `MCP_API_KEY` | No | — | Bearer token for HTTP auth |
| `MCP_HOST` | No | `0.0.0.0` | HTTP bind address |
| `MCP_PORT` | No | `8080` | HTTP port |
| `MCP_SAFETY_MODE` | No | `strict` | `strict` or `permissive` |
| `MCP_SAFETY_AUDIT` | No | — | `true` to log safety audit to stderr |
| `MCP_DEFAULT_CONTEXT` | No | — | JSON object merged into all contexts |
| `MCP_BOOTSTRAP_MODELS` | No | `res.partner,sale.order,account.move,product.product,stock.picking` | Models for session-bootstrap |

## Documentation

Full documentation in the **[Wiki](https://github.com/AlanOgic/odoo-mcp-19/wiki)**:

- [Getting Started](https://github.com/AlanOgic/odoo-mcp-19/wiki/Getting-Started)
- [Tools](https://github.com/AlanOgic/odoo-mcp-19/wiki/Tools)
- [Resources](https://github.com/AlanOgic/odoo-mcp-19/wiki/Resources)
- [ORM Methods](https://github.com/AlanOgic/odoo-mcp-19/wiki/ORM-Methods)
- [Module Knowledge](https://github.com/AlanOgic/odoo-mcp-19/wiki/Module-Knowledge)
- [AI Module](https://github.com/AlanOgic/odoo-mcp-19/wiki/AI-Module)
- [Domain Syntax](https://github.com/AlanOgic/odoo-mcp-19/wiki/Domain-Syntax)
- [Prompts](https://github.com/AlanOgic/odoo-mcp-19/wiki/Prompts)

## Requirements

- Python 3.10+
- Odoo 19+
- FastMCP 3.0.0b1+ (with tasks extra)

## License

MIT
