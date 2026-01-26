# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-01-26

### Added
- **Code-First Pattern** (98% token reduction)
  - `search_tools` - Search available operations by keyword on-demand
  - `discover_model_actions` - Discover all actions for a model dynamically
  - `execute_workflow` - Execute multi-step workflows in single calls
  - `TOOL_REGISTRY` - Pre-built operations for sales, accounting, CRM, stock, HR
- **Aggregation Support**
  - `aggregate_data` - Efficient `read_group` for reporting (sum, avg, min, max, count)
  - Date grouping support (:day, :week, :month, :quarter, :year)
- **Natural Language Model Discovery**
  - `find_model` tool - Translate business terms to Odoo models
  - `odoo://concepts` resource - Complete concept-to-model mappings
- **Rich Documentation Resources**
  - `odoo://model/{name}/docs` - Labels, help text, selection options from Odoo metadata
  - `odoo://docs/{model}` - Documentation URLs and GitHub links
  - `odoo://tool-registry` - List all pre-built workflows
- **Business Workflow Prompts**
  - `quote-to-cash` - Complete sales workflow
  - `ar-aging-report` - Accounts receivable aging
  - `inventory-check` - Stock levels analysis
  - `crm-pipeline` - Pipeline analysis
  - `customer-360` - Complete customer view
  - `daily-operations` - Operations dashboard
- **Module Knowledge Enhancements**
  - Added `mail` module with `message_type` documentation
  - Added `discuss` module with channel methods
  - Added `message_types` reference to API patterns
  - Added create examples for mail.message

### Fixed
- `arg_mapping.py`: Changed `create` mapping from `vals` to `vals_list` for v2 API

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

[1.1.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.1.0
[1.0.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.0.0
