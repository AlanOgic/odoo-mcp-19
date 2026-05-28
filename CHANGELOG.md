# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **`odoo://find-model/{concept}` resolves multi-word concepts** — natural-language
  phrases like `customer invoice` previously returned no match (the phrase was
  neither an exact alias nor a model substring). The resolver now tokenizes the
  concept and matches each word against the alias table, returning the union of
  known models (e.g. `res.partner` + `account.move`) with `source: "alias-token"`.
  Exact single- and multi-word aliases are unchanged.
- **Schema size hints are now relative, not absolute** — `quick-schema` descriptions
  (and the `_build_compact_schema` docstring / `read_resource` tool doc) dropped the
  misleading `~1.5KB` figure in favor of "~60-80% smaller than /fields", since the
  absolute size scales with the model's field count (e.g. `res.partner` ≈ 13 KB).

### Fixed
- **Schema resources leaked a cryptic error on missing/invalid models** — the
  `odoo://model/{m}/quick-schema`, `/fields`, `/schema` and `odoo://bundle`
  resources fed the `get_model_fields` error sentinel straight into the schema
  builders when a model was absent or its name malformed, surfacing the
  internal `'str' object has no attribute 'get'` instead of an actionable
  message. The resource path also skipped the model-name regex enforced on the
  tool path. A shared `_fetch_model_fields()` helper now validates the name via
  `_validate_model` and detects the sentinel before the builders run, returning
  a clear "model not found / invalid format" message with a lookup hint;
  `bundle` reports the bad model in its `errors` map with the same clean
  message. Covered by new `tests/test_resources.py` (6 unit tests, no live
  Odoo required).

## [1.14.0] - 2026-05-05

### Added
- **Token-based safety confirmation** — the safety gate now issues a single-use,
  120-second, operation-bound `confirmation_token` (cryptographic nonce) with each
  pending confirmation. The confirmation re-call must present a matching token via the
  new `confirmation_token` parameter; `confirmed=true` alone no longer bypasses the gate.
  Applied to all three gates: `execute_method`, `batch_execute`, `execute_workflow`.
- **Payload-digest token binding** — tokens are bound to a SHA-256 digest over the
  deterministic JSON of the operation payload (`_payload_digest()`), not just
  `(model, method)`. The confirmation re-call must reproduce the exact same args/kwargs
  seen at issue time, closing argument-substitution attacks (e.g. obtaining a token for
  `unlink([1])` then re-calling with `unlink([1..1000])`). Digest covers post-`resolve_json`,
  post-context-merge `{args, kwargs}` for `execute_method`, the full ops list for
  `batch_execute`, and the params dict for `execute_workflow`.
- **Verbose startup banner** — `_print_startup_banner()` prints a slant-ASCII startup
  card to **stderr** at boot (version, transport, masked credentials, safety mode, DX
  defaults, capability counts). Gated by `MCP_VERBOSE` (default `true`, both transports).
  Secrets masked to first 4 chars. No protocol changes (stdout stays clean).
- **Live integration tests** — `tests/live/test_safety_live.py` and
  `tests/live/test_v1110_live.py` (script-style runners, not pytest-collected).
- **Unit tests** — `tests/test_token_gate.py` covering happy path, single-use, TTL expiry,
  model/method/digest mismatch, and the argument-substitution attack vectors.

### Fixed
- **CRITICAL: `resolve_json` bypassed `BLOCKED_MODELS`** — the `target_model` in
  `resolve_json` was passed to `name_search` without validation, allowing reads from
  `ir.config_parameter`, `res.users`, `ir.rule`, etc. Now validated against the model
  regex **and** `BLOCKED_MODELS` before execution.
- **CRITICAL: stateless safety confirmation** — an agent could pass `confirmed=true` on
  the first call, skipping the gate entirely. Fixed by the token gate above.
- Read safety env vars (`MCP_SAFETY_MODE`, `MCP_SAFETY_AUDIT`, `MCP_DEFAULT_CONTEXT`) at
  call time so reconfiguration after import takes effect immediately.

### Changed
- **Blocked models expanded** — added `ir.model`, `ir.model.fields`, `res.groups`
  (now 8 total).
- Adopted the `logging` module for audit/diagnostic output (`odoo_mcp.safety` logger).

### Performance
- Faster resource dispatch and parallelized bundle/bootstrap fetches.

### Refactor
- **Split `server.py` (3762 → ~1226 lines) into 12 focused modules** with zero logic
  changes: `app.py`, `constants.py`, `models.py`, `utils.py`, `resources.py`,
  `prompts.py` extracted alongside the existing `server.py`, `odoo_client.py`,
  `arg_mapping.py`, `safety.py`, `__main__.py`. Import order matters — `app.py` binds the
  `mcp` decorator first so the resource/prompt/tool handlers register on import.

## [1.13.0] - 2026-04-10

### Added
- **Security hardening**: mandatory HTTP auth (`sys.exit(1)` if `MCP_API_KEY` unset in
  HTTP mode), input validation (model/method regex, URI-scheme guard), sanitized error
  responses (Odoo `debug` tracebacks never forwarded to clients), non-root Docker user
  (UID 1001), `--env-file` for secrets.
- **Thread safety**: singleton `OdooClient` with double-checked locking; `threading.Lock`
  on `_DOC_CACHE` (100-entry LRU) and `RUNTIME_MODEL_ISSUES`.
- **Resource limits**: `MCP_DEFAULT_CONTEXT` capped at 4 KB; `MCP_BOOTSTRAP_MODELS` capped
  at 20 models.

### Fixed
- `ValueError` self-catch in error parsing.
- Unbound `method_info` referenced outside its `for` loop.
- `batch_execute` missing type checks on `args`/`kwargs`.
- Improved error guidance for relational writes, dates, and state changes.
- Skip the `X-Odoo-Database` header for Odoo.com SaaS instances (#1).

### Changed
- **Dependencies**: `fastmcp>=3.2.0`, `requests>=2.32.4` (CVE fix).

## [1.12.0] - 2026-03-10

### Changed
- **FastMCP SDK upgrade 3.0.0b1 → 3.1.0** — providers, transforms, component versioning,
  OpenTelemetry, and MultiAuth support.

### Added
- HTTP transport in Docker Compose, setup wizard coverage, and docs.

### Fixed
- Inconsistent lifespan context access in `execute_method` — now uses
  `get_odoo_client()`.

## [1.11.0] - 2026-02-12

### Added
- `odoo://model/{model}/quick-schema` — ultra-compact schema (~1.5 KB, 60–80% smaller,
  short keys `t`/`req`/`ro`/`rel`).
- `odoo://bundle/{models}` — batch quick-schema for up to 10 models in one call.
- `odoo://session-bootstrap` — one-call kickoff with schemas + workflows for
  `MCP_BOOTSTRAP_MODELS`.
- `odoo://model/{model}/workflow` — state machines for 6 models with transitions and
  side effects.
- `resolve_json` parameter on `execute_method` for Many2one name→ID auto-resolution.
- `MCP_DEFAULT_CONTEXT` env var merged into all operation contexts.

### Changed
- Expanded error patterns from ~5 to ~25, with `{model}` template substitution.

### Fixed
- Preserve context through the `search_read` → `search`+`read` fallback.

## [1.10.0] - 2026-02-11

### Added
- **Pre-execution safety classification layer** (`safety.py`, zero FastMCP dependency) —
  classifies every operation as `SAFE` / `MEDIUM` / `HIGH` / `BLOCKED` before execution.
  - **Blocked models** (writes refused): `ir.rule`, `ir.model.access`, `ir.module.module`,
    `ir.config_parameter`, `res.users`.
  - **Sensitive models** (writes confirm): `account.move`, `account.payment`,
    `account.bank.statement`, `hr.payslip`, `ir.cron`.
  - **Cascade warnings** for known side effects (`action_confirm`, `action_post`,
    `button_validate`).
  - `confirmed=true` parameter on `execute_method`, `batch_execute`, `execute_workflow`.
  - Audit logging to stderr via `MCP_SAFETY_AUDIT=true`.
- `read_resource` tool — bridge for clients (e.g. Claude Desktop) without resource-template
  support.
- Live `/doc-bearer/` enrichment for `odoo://methods/{model}` (signatures, return types,
  decorators) with static fallback.
- Lightweight `odoo://model/{model}/fields` resource and improved resource discoverability.

## [1.9.0] - 2026-02-02

### Added
- **MCP 2025-11-25 Specification Features**
  - **Background Tasks (SEP-1686)**: `batch_execute` and `execute_workflow` now support async execution with progress tracking via `task=True` and `Progress` dependency
  - **Icons (SEP-973)**: Server and all tools now include Odoo brand icon (#714B67) for visual recognition in MCP clients
  - **Structured Output Schemas**: All tools return typed Pydantic response models instead of plain dicts
  - **User Elicitation**: New `configure_odoo` tool for interactive connection configuration

- **New Tool**: `configure_odoo`
  - Interactive connection setup using MCP elicitation protocol
  - Collects URL, database, authentication method, and username
  - Returns environment variable instructions
  - Graceful fallback when elicitation not supported by client

- **Enhanced Response Models**
  - `ExecuteMethodResponse`: Added `IssueAnalysis`, `execution_time_ms`, `fallback_used` fields
  - `BatchExecuteResponse`: Added `BatchOperationResult` list, `execution_time_ms`
  - `ExecuteWorkflowResponse`: New model with `WorkflowStepResult` list, workflow-specific fields
  - All responses now include execution timing for performance monitoring

- **New Assets**
  - `assets/odoo_icon.svg`: Odoo brand icon in SVG format

### Changed
- **Dependencies**: Updated to `fastmcp[tasks]>=3.0.0b1` for background task support
- **Tools count**: 4 tools (was 3) - added `configure_odoo`
- `batch_execute`: Now async with progress tracking
- `execute_workflow`: Now async with progress tracking

### Technical
- Removed `Context` dependency from task-enabled tools (uses `get_odoo_client()` directly)
- Added `asyncio` import for background task coordination
- Added `time` import for execution timing
- Icon loaded as base64 data URI at module initialization

## [1.8.0] - 2026-01-30

### Added
- **HTTP Transport** with Bearer token authentication
  - New `MCP_TRANSPORT` env var: `stdio` (default) or `streamable-http`
  - New `MCP_API_KEY` env var for Bearer token authentication
  - New `MCP_HOST` and `MCP_PORT` env vars for HTTP server configuration
  - Enables remote MCP server deployments

### Changed
- Added `StaticTokenVerifier` authentication provider for HTTP transport
- Updated documentation with HTTP transport examples

## [1.7.0] - 2026-01-29

### Added
- **Automatic search_read Fallback** with error categorization
  - When `search_read` fails with 500 error, automatically falls back to `search` + `read`
  - Error categorization: timeout, relational_filter, computed_field, access_rights, memory, data_integrity
  - Domain pattern detection: dot notation, complex OR, negation, any operator, hierarchical
  - Problematic field detection for known models (e.g., `stock.move.line`)
  - Runtime issue tracking: Builds knowledge base of model/method problems during operation

- **Model Limitations Resource** (`odoo://model-limitations`)
  - Static limitations from source code analysis
  - Runtime-detected issues with categorization
  - Pattern summary across all models
  - Global recommendations based on detected patterns

- **Known Limitations: stock.move.line**
  - `picking_type_id` with `!=` operator causes NotImplemented error
  - `lots_visible` field is non-stored computed
  - `product_category_name` is 3-level deep related field
  - Documented safe fields and fields to avoid

### Changed
- `execute_method` now returns detailed `issue_analysis` when fallback is used
- Enhanced error messages with suggested solutions

## [1.6.0] - 2026-01-28

### Added
- **AI Module Documentation** (Enterprise) - Complete documentation for Odoo 19 AI module
  - 6 models documented: `ai.agent`, `ai.topic`, `ai.agent.source`, `ai.embedding`, `ai.composer`, `ai.prompt.button`
  - 5 special methods: `get_direct_response`, `open_agent_chat`, `create_from_attachments`, `create_from_binary_files`, `create_from_urls`
  - LLM models reference: OpenAI (gpt-3.5-turbo through gpt-5), Google (gemini-2.5-pro/flash)
  - Response styles: analytical (0.2), balanced (0.5), creative (0.8)
  - Default topics: Natural Language Search, Information Retrieval, Create Leads
  - RAG workflow documentation (source → process → embed → search)
  - AI Tools system (ir.actions.server with use_in_ai=True)

- **16 New ORM Methods** documented in `orm_methods` section
  - Recordset operations: `browse`, `exists`, `ensure_one`
  - Filtering/mapping: `filtered`, `filtered_domain`, `mapped`, `sorted`, `grouped`
  - Environment modifiers: `with_context`, `with_user`, `with_company`, `sudo`
  - Creation: `name_create`
  - Fetching: `search_fetch`, `fetch`
  - Archiving: `action_archive`, `action_unarchive`

- **Enhanced Module Documentation** (validated against Odoo 19 source code)
  - `sale.order`: +7 methods (action_lock, action_unlock, action_quotation_sent, action_preview_sale_order, action_update_taxes, action_update_prices, action_view_invoice)
  - `account.move`: +8 methods (button_set_checked, action_force_register_payment, action_reverse, action_duplicate, action_send_and_print, action_invoice_sent, action_switch_move_type) + move_types reference
  - `crm.lead`: +6 methods (action_set_won_rainbowman, action_set_automated_probability, action_reschedule_meeting, action_show_potential_duplicates, action_restore) + lead vs opportunity docs
  - `stock.picking`: +9 methods (action_detailed_operations, action_split_transfer, action_toggle_is_locked, action_put_in_pack, button_scrap, action_see_move_scrap, action_see_packages, action_see_returns, action_view_reception_report)
  - `purchase.order`: +6 methods (button_approve, button_lock, button_unlock, action_rfq_send, action_acknowledge, action_merge)

- **Relational Commands Documentation** - Enhanced Command class reference with CREATE, UPDATE, DELETE, UNLINK, LINK, CLEAR, SET

### Changed
- **module_knowledge.json** upgraded to v1.4.0
- Added `_validated_against` field for traceability
- Total modules: 13 (was 12)
- Total ORM methods: 30 (was 14)

### Documentation
- Validated against Odoo 19 Enterprise source code
- Validated against official Odoo 19 documentation
- Source paths: `/odoo-19.0+e.20260128/` (Enterprise), `/odoo-19.0.post20260128/` (Community)

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

[1.9.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.9.0
[1.8.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.8.0
[1.7.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.7.0
[1.6.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.6.0
[1.5.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.5.0
[1.4.1]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.4.1
[1.4.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.4.0
[1.3.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.3.0
[1.2.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.2.0
[1.1.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.1.0
[1.0.0]: https://github.com/AlanOgic/odoo-mcp-19/releases/tag/v1.0.0
