# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-26

### Added
- Initial release of Odoo MCP Server for Odoo 19+
- **Tools**
  - `execute_method` - Execute any Odoo method on any model
  - `batch_execute` - Execute multiple operations atomically
  - `get_model_methods` - Discover available methods on a model
- **Resources**
  - `odoo://models` - List all models
  - `odoo://model/{name}` - Model info with fields
  - `odoo://model/{name}/schema` - Complete schema
  - `odoo://record/{model}/{id}` - Get specific record
  - `odoo://methods/{model}` - Available methods
  - `odoo://workflows` - Business workflows
  - `odoo://server/info` - Server information
  - `odoo://module-knowledge` - All module-specific knowledge
  - `odoo://module-knowledge/{name}` - Knowledge for specific module
- **Module Knowledge Base**
  - 10 Odoo modules: knowledge, sale, account, crm, stock, purchase, hr_expense, hr_leave, project, documents
  - Special methods documentation for each module
  - Error patterns with contextual suggestions
- **Infrastructure**
  - v2 JSON-2 API only (Bearer token authentication)
  - Argument mapping for positional to named args conversion
  - Smart limits (DEFAULT_LIMIT=100, MAX_LIMIT=1000)
  - Docker support with `Dockerfile` and `docker-compose.yml`
  - `run-docker.sh` wrapper for Claude Desktop

[1.0.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.0.0
