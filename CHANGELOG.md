# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-01-28

### Added
- **Official JSON-2 API Specification** from Odoo 19.0 documentation
  - Endpoint format: `/json/2/{model}/{method}`
  - Authentication: Bearer token via `Authorization` header
  - Database: `X-Odoo-Database` header
  - Transaction behavior documented (each call = own SQL transaction)
  - Deprecation notice: XML-RPC/JSON-RPC removed in Odoo 20 (fall 2026)
  - New `/doc` endpoint for built-in API documentation

- **Comprehensive ORM Method Documentation** (`orm_methods` section)
  - `search()` - with `count` parameter for lightweight counting
  - `search_count()` - lightest option for totals
  - `search_read()` - with `load` parameter for Many2one control
  - `read()` - with `load` parameter ('_classic_read' vs null)
  - `create()` - batch creation, @api.model_create_multi, relational commands
  - `write()` - all relational field commands (0-6 tuples)
  - `unlink()` - cascade behavior, ondelete options
  - `name_search()` - autocomplete search for Many2one
  - `default_get()` - get default values for new records
  - `fields_get()` - schema introspection
  - `copy()` - record duplication with overrides
  - `check_access_rights()` - permission checking

- **Field Types Reference** (`field_types` section)
  - Simple types: boolean, char, text, integer, float, date, datetime, binary, html, monetary
  - Selection types: selection, reference
  - Relational types: many2one, one2many, many2many with JSON read/write formats
  - Common field attributes documented

- **Method Decorators Reference** (`method_decorators` section)
  - RPC-accessible: @api.model, @api.model_create_multi, @api.readonly
  - NOT RPC-accessible: @api.private, @api.onchange
  - Validation decorators: @api.constrains, @api.ondelete
  - Computed field decorators: @api.depends, @api.depends_context

- **Model Types Reference** (`model_types` section)
  - models.Model - persistent with DB table
  - models.TransientModel - temporary, auto-cleanup (wizards)
  - models.AbstractModel - no table, for mixins

- **Model Attributes Reference** (`model_attributes` section)
  - _name, _description, _order, _rec_name
  - _inherit, _inherits (inheritance patterns)
  - _parent_name, _parent_store (hierarchies)
  - _sql_constraints, _log_access

- **Enhanced `odoo://methods/{model}` Resource**
  - Now returns 4 categories: read_methods, write_methods, introspection_methods, special_methods
  - Added read_group to read_methods
  - Added copy to write_methods
  - New introspection_methods category

### Changed
- **module_knowledge.json** upgraded to v1.3.0
- Added `load` parameter to arg_mapping.py for read/search_read methods
- Added `lazy` parameter documentation to aggregation section
- Enhanced method descriptions in server.py with parameter details

### Documentation
- Updated from Cybrosys Odoo 19 technical articles and official Odoo documentation
- Sources: odoo.com/documentation/19.0, cybrosys.com/blog (10+ articles reviewed)

## [1.4.1] - 2026-01-27

### Changed
- **Best Practice Enforcement**: Schema introspection is now MANDATORY before queries
  - Updated `execute_method` tool description with "MANDATORY WORKFLOW" section
  - Added clear guidance: "Never guess field names - introspect schema first"
  - Prevents wasted API calls from guessing field names based on "common patterns"

### Documentation
- Updated CLAUDE.md Golden Rule: "Schema First, Query Second"
- Updated Pre-Execution Checklist with schema introspection as step 1
- Added explanation: "Guessing field names based on 'common patterns' wastes API calls"

## [1.4.0] - 2026-01-27

### Changed
- **BREAKING**: Simplified to only 3 tools (was 8)
  - Kept: `execute_method`, `batch_execute`, `execute_workflow`
  - Removed: `get_model_methods`, `find_model`, `search_tools`, `discover_model_actions`, `aggregate_data`
- All discovery functionality moved to resources for better token efficiency

### Added
- **New Resources for Discovery:**
  - `odoo://find-model/{concept}` - Natural language to model name lookup
  - `odoo://tools/{query}` - Search available operations by keyword
  - `odoo://actions/{model}` - Discover all actions for a model
- Updated all prompts to use resources instead of removed tools

### Removed
- `get_model_methods` tool (redundant with `odoo://methods/{model}` resource)
- `find_model` tool (now `odoo://find-model/{concept}` resource)
- `search_tools` tool (now `odoo://tools/{query}` resource)
- `discover_model_actions` tool (now `odoo://actions/{model}` resource)
- `aggregate_data` tool (use `execute_method` with `read_group` + `odoo://aggregation` guide)

### Why This Change?
Resources are read-only discovery mechanisms that don't count as "tools" in the MCP context.
This reduces cognitive load and keeps the tool interface minimal:
- **Tools** = Actions that modify data or execute operations
- **Resources** = Read-only discovery and documentation

## [1.3.0] - 2026-01-27

### Added
- **Domain Syntax Resource** (`odoo://domain-syntax`)
  - Complete operator reference: comparison, list, text, hierarchical, relational, logic
  - New operators documented: `child_of`, `parent_of`, `=?`, `any`, `not any`, `not =like`, `not =ilike`
  - Dot notation examples for related field traversal
  - Complex domain examples with Polish notation logic
- **Pagination Resource** (`odoo://pagination`)
  - Guide for offset/limit parameters
  - Pattern for getting total count with `search_count`
  - Batch iteration pattern for large datasets
- **Hierarchical Query Resource** (`odoo://hierarchical`)
  - Parent/child tree traversal patterns
  - Common hierarchical models documented (product.category, hr.department, stock.location, etc.)
  - Query patterns: descendants, ancestors, direct children, root records, tree path
- **Aggregation Resource** (`odoo://aggregation`)
  - Complete `read_group` reference with all aggregators
  - Date grouping syntax (`:day`, `:week`, `:month`, `:quarter`, `:year`)
  - Examples for common reporting scenarios
  - v19 deprecation note (use `formatted_read_group` for new code)
- **New MCP Prompts**
  - `domain-builder` - Interactive domain filter construction
  - `hierarchical-query` - Guide for parent/child queries with examples
  - `paginated-search` - Pagination pattern with total count
  - `aggregation-report` - read_group reporting guide

### Changed
- **module_knowledge.json** enhanced to v1.2.0
  - Added `domain_syntax` section with all operators and examples
  - Added `pagination` section with patterns
  - Added `hierarchical_models` section with common models and query patterns
  - Added `aggregation` section with read_group reference and v19 notes

### Documentation
- Updated CLAUDE.md with domain operators quick reference
- Added hierarchical queries section to CLAUDE.md
- Added pagination pattern section to CLAUDE.md

## [1.2.0] - 2026-01-27

### Changed
- **BREAKING**: Upgraded to FastMCP 3.0.0 (from 2.13.0+)
  - Updated all prompts to use `Message` class from `fastmcp.prompts`
  - Prompts now return `list[Message]` instead of `List[Dict[str, str]]`
  - Better compatibility with MCP ecosystem and future updates
- Pinned FastMCP to `>=3.0.0,<4` for stability

### Security
- FastMCP 3.0 addresses security vulnerabilities from 2.x versions
- Improved authentication architecture for HTTP transport deployments

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

[1.5.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.5.0
[1.4.1]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.4.1
[1.4.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.4.0
[1.3.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.3.0
[1.2.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.2.0
[1.1.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.1.0
[1.0.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.0.0
