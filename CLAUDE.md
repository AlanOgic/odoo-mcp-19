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
- 3 tools: `execute_method`, `batch_execute`, `get_model_methods`
- 10+ resources for discovery
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
