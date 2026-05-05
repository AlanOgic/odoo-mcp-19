```
   ____      __               __  _____________     _______
  / __ \____/ /___  ____     /  |/  / ____/ __ \   <  / __ \  __
 / / / / __  / __ \/ __ \   / /|_/ / /   / /_/ /   / / /_/ /_/ /_
/ /_/ / /_/ / /_/ / /_/ /  / /  / / /___/ ____/   / /\__, /_  __/
\____/\__,_/\____/\____/  /_/  /_/\____/_/       /_//____/ /_/
```

Ask Claude to read and write your Odoo 19 data in plain language — and stop it from doing
the dangerous parts by accident.

This server connects any Model Context Protocol (MCP) client to Odoo 19+ over the v2 JSON-2
API. An assistant discovers your models by reading schemas rather than guessing field names,
then calls any ORM method through one tool. Before anything destructive runs, a safety layer
classifies the operation and holds it behind a single-use confirmation token.

```python
# The assistant reads the schema first, then queries — no guessed field names
read_resource("odoo://model/sale.order/quick-schema")
execute_method("sale.order", "search_read",
    kwargs_json='{"domain": [["state", "=", "sale"]], "fields": ["name", "amount_total"], "limit": 10}')
```

## Who this is for

- **You run Odoo 19+** and want an assistant that queries and updates it directly, instead of
  copying data between a chat window and the ERP.
- **You need writes to be safe.** Posting a journal entry or validating a picking is
  irreversible. Every such call is gated, and eight security-critical models refuse writes
  outright.
- **You care about token cost.** Compact schemas, batched model bundles, and one-call session
  bootstrap keep discovery cheap on long conversations.

If you only need read-only reporting, this still works — set a `readonly` role and the safety
layer blocks every non-safe method.

## What you get

| | |
|---|---|
| **5 tools** | `execute_method` reaches any method on any model; `batch_execute`, `execute_workflow`, `configure_odoo`, and `read_resource` cover the rest |
| **28 resources** | Model discovery, compact schemas, state-machine workflows, introspection, and runtime posture (`odoo://server-status`) |
| **19 prompts** | 12 generic guided workflows plus 7 `cyanview-*` skill prompts, gated per user in multi-user mode |
| **Safety layer** | Risk classification before execution, 8 blocked models, 6 sensitive models, cascade warnings |
| **Locked mode** | One flag (`MCP_SAFETY_MODE=locked`) turns writes off, enforces a write allowlist, binds HTTP to localhost, and pre-flights payloads against live `fields_get` — each layer independently overridable |
| **Multi-user mode** | Per-user bearer keys and personal Odoo clients, so every write is attributed to a real person |
| **Reference data** | 30 documented ORM methods, 13 modules with special-method knowledge, including the Enterprise AI module |

Built on **MCP 2025-11-25** (background tasks, progress tracking, icons, structured outputs)
and **FastMCP `>=3.4.6,<4`**. The ceiling is deliberate — FastMCP 4.x targets MCP spec
2026-07-28 and is [not yet adopted](docs/mcp-2026-07-28-migration.md).

Hardened by default: regex-validated model and method names, a non-root Docker container,
mandatory authentication on HTTP, thread-safe caches, and no traceback ever forwarded to a
client.

## Installation

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (recommended) **OR** Python 3.10+
- An Odoo 19+ instance with API access
- An API key from your Odoo instance (**Preferences → Account Security → New API Key**)

### Option A: Docker (recommended)

**Build from source:**

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

### Option C: pip install (from GitHub)

```bash
pip install git+https://github.com/AlanOgic/odoo-mcp-19.git
```

## Setup wizard

Interactive wizard that generates `.env`, Docker commands, and Claude Desktop config:

```bash
# From source
python -m odoo_mcp --setup

# From pip
odoo-mcp-19 --setup
```

The wizard walks you through Odoo connection, transport (stdio or streamable-http), safety mode, and outputs ready-to-use configuration files.

## Configure Claude Desktop

Edit your Claude Desktop configuration file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

> **Server name:** register this server as `odoo19-mcp` (the name the `--setup`
> wizard generates and the one used in every example below). The companion
> Cyanview skills assume this name in their `allowed-tools`, so a different name
> means those skills won't be permitted to call the server's tools.

### Using Docker (recommended)

**Method 1: Using `run-docker.sh` wrapper (simplest)**

Create a `.env` file in the project root with your credentials, then:

```json
{
  "mcpServers": {
    "odoo19-mcp": {
      "command": "/path/to/odoo-mcp-19/run-docker.sh"
    }
  }
}
```

**Method 2: Inline Docker command**

```json
{
  "mcpServers": {
    "odoo19-mcp": {
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
    "odoo19-mcp": {
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
    "odoo19-mcp": {
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

### Verify the connection

Ask Claude: *"List the first 5 partners in Odoo"*. It should call `execute_method` on
`res.partner` and come back with names.

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

# Confirm a sale order (2-step safety confirmation with token)
# Step 1: triggers safety gate, returns confirmation_token in hint
result = execute_method("sale.order", "action_confirm", args_json='[[15]]')
# Step 2: confirm with the token from step 1
execute_method("sale.order", "action_confirm", args_json='[[15]]',
    confirmed=true, confirmation_token='<token from step 1>')

# Multi-step workflow in one call (also gated — posting an invoice is irreversible)
execute_workflow("create_and_post_invoice", '{"partner_id": 123, "invoice_lines": [...]}')
```

Note that `confirmed=true` on its own does nothing. The token is single-use, expires after
120 seconds, and is bound to the exact payload the gate inspected — so an agent cannot take a
token issued for `unlink([1])` and reuse it on `unlink([1, 2, …, 1000])`.

## Architecture

### Tools (5)

| Tool | Purpose |
|------|---------|
| `execute_method` | Call any method on any Odoo model |
| `batch_execute` | Multiple operations with progress tracking |
| `execute_workflow` | Pre-built multi-step workflows |
| `configure_odoo` | Interactive connection setup |
| `read_resource` | Read any `odoo://` resource by URI |

### Resources (28)

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
| `odoo://server-status` | Runtime safety posture (mode, host, allowlist, warnings). Non-secret. **New in v1.15.0.** |

### Prompts (19)

12 generic guided prompts:

| Prompt | Purpose |
|--------|---------|
| `odoo-exploration` | Discover instance capabilities |
| `search-records` | Search for records in a model |
| `odoo-api-reference` | Quick API reference card |
| `ar-aging-report` | Accounts receivable aging |
| `inventory-check` | Stock levels analysis |
| `crm-pipeline` | Pipeline analysis |
| `customer-360` | Complete customer view |
| `daily-operations` | Operations dashboard |
| `domain-builder` | Build complex domain filters |
| `hierarchical-query` | Query parent/child trees |
| `paginated-search` | Paginate large result sets |
| `aggregation-report` | Aggregation reports |

7 `cyanview-*` workflow skill prompts (bodies loaded from `skills/*.md`; in multi-user mode each is gated per user via the `user_skills` allowlist):

| Prompt | Purpose |
|--------|---------|
| `cyanview-quote` | Build a Cyanview sales quotation |
| `cyanview-rma` | Manage an RMA / repair order |
| `cyanview-customer-360` | Full 360° customer briefing |
| `cyanview-serial-tracker` | Trace a device by serial number |
| `cyanview-project-designer` | Design a camera-control system |
| `cyanview-shipping-watchdog` | Audit unshipped/overdue orders |
| `cyanview-inventory-watchdog` | Monitor stock levels and reorders |

## Safety layer

Pre-execution safety classification gates dangerous operations behind confirmation.

### Risk levels

| Level | Behavior | Confirm? |
|-------|----------|----------|
| `SAFE` | Execute immediately | Never |
| `MEDIUM` | Gate based on mode/volume | Conditional |
| `HIGH` | Always require confirmation | Always |
| `BLOCKED` | Always refuse | N/A |

### Blocked models (write always refused)

`ir.rule`, `ir.model.access`, `ir.module.module`, `ir.config_parameter`, `ir.model`, `res.users`, `res.groups`, `res.users.apikeys`

### Sensitive models (write always confirms)

`account.move`, `account.payment`, `account.bank.statement`, `hr.payslip`, `ir.cron`, `ir.model.fields`

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
3. If confirmation needed: returns `pending_confirmation=true` with `safety` classification and a `confirmation_token` in the `hint` field
4. Caller reviews, then re-calls with `confirmed=true` AND `confirmation_token='<token>'`

Tokens are single-use, expire after 120s, and are bound to the specific model+method. This prevents agents from bypassing the safety gate by always passing `confirmed=true`.

### Locked mode (v1.15.0)

`MCP_SAFETY_MODE=locked` adds four overlapping protections on top of the classifier and token gate. Designed for safe-by-default deployments where a colleague might point an over-eager AI agent at their own Odoo instance.

| Protection | Default under `locked` | Override env var |
|---|---|---|
| Global write kill-switch | on | `MCP_READ_ONLY=false` |
| Side-effect allowlist enforced | on (empty list = no writes) | populate `MCP_WRITE_ALLOWLIST` |
| HTTP bind | `127.0.0.1` (localhost-only) | `MCP_HOST=0.0.0.0` |
| Live `fields_get` payload pre-flight | on | `MCP_VALIDATE_PAYLOADS=false` |

Each protection is independently overridable — set `locked` and tune the individual flags as needed. `permissive` and `strict` semantics are unchanged from v1.14.0; no existing deploy changes posture on upgrade unless the operator opts in.

**Allowlist syntax**: `MCP_WRITE_ALLOWLIST="sale.order.action_confirm,res.partner.message_post,product.product.*"`. Each entry is `model.method` or `model.*` (any method on the model). Cross-model wildcards (`*.method`) are rejected — too easy to over-grant.

**Three typical configurations:**

```bash
# Safe-by-default — paste this and forget
MCP_SAFETY_MODE=locked

# Locked but allow specific side-effect methods
MCP_SAFETY_MODE=locked
MCP_READ_ONLY=false
MCP_WRITE_ALLOWLIST=res.partner.message_post,sale.order.action_confirm

# Power user — current v1.14.0 behaviour, no changes
MCP_SAFETY_MODE=strict
```

**Posture introspection**: read `odoo://server-status` to see the resolved profile at runtime — current mode, host, allowlist contents, and any foot-gun warnings (e.g. `locked` + `MCP_HOST=0.0.0.0`, or `locked` + `MCP_READ_ONLY=false`). The startup banner shows a one-line summary: `[SAFETY locked · READ-ONLY · BIND 127.0.0.1 · ALLOWLIST 0 entries]`.

**Payload pre-flight** (`MCP_VALIDATE_PAYLOADS=true`): before issuing a confirmation token for a write, the validator fetches `fields_get` for the target model (60s LRU cache) and rejects payloads that reference non-existent fields, write to readonly fields, or arrive when the connection is silently down (`fields_get` returns `{}`). Catches hallucinated field names before they reach Odoo.

## DX Improvements (v1.11.0)
## Token-efficient discovery

### Quick schema

`odoo://model/{model}/quick-schema` — Ultra-compact schema with short keys: `t` (type), `req` (required), `ro` (readonly), `rel` (relation). ~60-80% smaller than `/fields`.

### Bundle

`odoo://bundle/res.partner,sale.order,stock.picking` — Batch quick-schema for up to 10 models in one call.

### Session bootstrap

`odoo://session-bootstrap` — One call to bootstrap a conversation with schemas + workflows for common models. Configure via `MCP_BOOTSTRAP_MODELS` env var.

### Workflow

`odoo://model/{model}/workflow` — State machine transitions for 6 main models with side effects and irreversibility flags. Dynamic fallback for unmapped models.

### Many2one resolution

Auto-resolve field names to IDs with `resolve_json`:

```python
execute_method("res.partner", "write",
    args_json='[[1], {"user_id": null}]',
    resolve_json='{"user_id": {"model": "res.users", "search": "John"}}')
```

### Default context

Set `MCP_DEFAULT_CONTEXT` to apply context to all operations:

```bash
export MCP_DEFAULT_CONTEXT='{"lang": "fr_FR", "tz": "Europe/Paris"}'
```

### Error suggestions

~25 error patterns with actionable suggestions covering 422, 500, 403, 404 errors and fallback patterns.

## HTTP transport

Run as an HTTP server with Bearer token authentication for remote or multi-client access.

### Docker Compose (recommended)

Add to your `.env`:

```bash
MCP_TRANSPORT=streamable-http
MCP_API_KEY=your-secret-bearer-token
```

Then:

```bash
docker compose up -d
```

The MCP endpoint is available at `http://localhost:8080/mcp`.

### Docker run

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

### Claude Desktop config (HTTP)

```json
{
  "mcpServers": {
    "odoo19-mcp": {
      "type": "streamable-http",
      "url": "http://localhost:8080/mcp",
      "headers": {
        "Authorization": "Bearer your-secret-token"
      }
    }
  }
}
```

### Verify authentication

```bash
# Should succeed (200)
curl -i -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "initialize", "id": 1}'

# Should fail (401/403) — wrong or missing token
curl -i -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "initialize", "id": 1}'
```

## Multi-user mode

The HTTP transport above serves a single Odoo account behind one static `MCP_API_KEY`.
For teams, set **`USERS_DB_PATH`** to switch into multi-user mode: many users share one
server process, each authenticating with their own bearer token and acting as their **own**
Odoo account — so every write is attributed to the real person, not a shared service user.

### How it works

- **Registry** — users, their per-server API keys, encrypted Odoo credentials, and skill
  allowlists live in a SQLite registry (`users.db`) owned and written by **CLORAG** (a
  companion app with an `/admin/users` page). This server is a **pure reader** — it opens
  the database with `mode=ro` and never writes.
- **Per-request identity** — an incoming bearer token is hashed (sha256) and looked up in
  the registry. The caller gets a **personal `OdooClient`** built from their stored Odoo
  username + decrypted API key (cached 300s, re-checked on credential rotation). Plaintext
  keys are never stored; the static `MCP_API_KEY` still works and maps to an `env-admin`
  identity.
- **Roles** — `admin` → unrestricted; `readonly` → read-only, with the safety layer blocking
  every non-safe method; other roles → normal safety rules on their own account.
- **Per-user skills** — the `cyanview-*` workflow prompts are filtered per user against a
  `user_skills` allowlist (fails closed without a token). Generic prompts stay visible to all.

### Deployment

```bash
# Requires: USERS_DB_PATH, TOKEN_ENCRYPTION_KEY (must equal CLORAG's), MCP_TRANSPORT=streamable-http
docker compose -f docker-compose.yml -f docker-compose.multiuser.yml up -d
```

The overlay mounts the CLORAG data dir read-only at `/registry` and injects the encryption
key as a Docker secret. The `TOKEN_ENCRYPTION_KEY` and the `.token_salt` file must be
**identical** to CLORAG's, or stored credentials cannot be decrypted. In multi-user mode the
env `ODOO_*` credentials become optional (only the `env-admin` fallback uses them), though
`ODOO_URL` / `ODOO_DB` are still needed to build per-user clients.

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ODOO_URL` | Yes | — | Odoo server URL |
| `ODOO_DB` | Yes | — | Database name |
| `ODOO_USERNAME` | Yes | — | Username |
| `ODOO_API_KEY` | Yes | — | API key (Preferences → Account Security) |
| `ODOO_PASSWORD` | No | — | Password (fallback if no API key) |
| `ODOO_TIMEOUT` | No | `30` | Request timeout in seconds |
| `ODOO_VERIFY_SSL` | No | `true` | SSL certificate verification |
| `MCP_TRANSPORT` | No | `stdio` | Transport: `stdio` or `streamable-http` |
| `MCP_API_KEY` | HTTP: this **or** `USERS_DB_PATH` | — | Static bearer token (single-user HTTP, or admin fallback in multi-user mode). HTTP server `sys.exit(1)` if neither is set |
| `USERS_DB_PATH` | No | — | Path to the CLORAG registry (`users.db`) → enables [multi-user mode](#multi-user-mode) |
| `TOKEN_ENCRYPTION_KEY` / `TOKEN_ENCRYPTION_KEY_FILE` | With `USERS_DB_PATH` | — | Secret to decrypt registry Odoo credentials — must equal CLORAG's value |
| `MCP_HOST` | No | `0.0.0.0` (`127.0.0.1` under `locked`) | HTTP bind address. Default flips to localhost when `MCP_SAFETY_MODE=locked`. |
| `MCP_PORT` | No | `8080` | HTTP port |
| `MCP_VERBOSE` | No | `true` | Slant-ASCII startup banner to stderr (version, transport, masked creds, safety mode, capability counts). `false` to silence |
| `MCP_SAFETY_MODE` | No | `strict` | `permissive`, `strict`, or **`locked`** (v1.15.0). `locked` activates `MCP_READ_ONLY=true`, `MCP_WRITE_ALLOWLIST` enforcement, `MCP_HOST=127.0.0.1` default, and `MCP_VALIDATE_PAYLOADS=true`. |
| `MCP_SAFETY_MODE` | No | `strict` | `permissive`, `strict`, or **`locked`** (v1.15.0). `locked` activates `MCP_READ_ONLY=true`, `MCP_WRITE_ALLOWLIST` enforcement, `MCP_HOST=127.0.0.1` default, and `MCP_VALIDATE_PAYLOADS=true`. |
| `MCP_READ_ONLY` | No | derived from mode | `true` to globally reject all side-effect methods (`create`, `write`, `unlink`, `copy`, `name_create`, `load`, `action_*`, `button_*`, `_action_*`). Reads pass through. |
| `MCP_WRITE_ALLOWLIST` | No | empty | Comma-separated `model.method` (or `model.*`) entries permitted as side effects. Enforced under `locked` mode, or whenever set explicitly. |
| `MCP_VALIDATE_PAYLOADS` | No | derived from mode | `true` to validate write payloads against live `fields_get` before issuing a confirmation token. Catches hallucinated fields and readonly writes. |
| `MCP_SAFETY_AUDIT` | No | — | `true` to log safety audit to stderr |
| `MCP_DEFAULT_CONTEXT` | No | — | JSON object merged into all contexts (max 4KB) |
| `MCP_BOOTSTRAP_MODELS` | No | `res.partner,sale.order,account.move,product.product,stock.picking` | Models for session-bootstrap (max 20) |

## Documentation

Full documentation lives in the **[Wiki](https://github.com/AlanOgic/odoo-mcp-19/wiki)**:

**Start here**

- [Getting started](https://github.com/AlanOgic/odoo-mcp-19/wiki/Getting-Started) — install and connect in five minutes
- [Tools](https://github.com/AlanOgic/odoo-mcp-19/wiki/Tools) — the 5 tools, their parameters, and when to reach for each
- [Resources](https://github.com/AlanOgic/odoo-mcp-19/wiki/Resources) — the 27 `odoo://` discovery URIs

**Reference**

- [ORM methods](https://github.com/AlanOgic/odoo-mcp-19/wiki/ORM-Methods) — 30 methods with working examples
- [Domain syntax](https://github.com/AlanOgic/odoo-mcp-19/wiki/Domain-Syntax) — Polish-prefix search filters
- [Module knowledge](https://github.com/AlanOgic/odoo-mcp-19/wiki/Module-Knowledge) — special methods across 13 modules
- [AI module](https://github.com/AlanOgic/odoo-mcp-19/wiki/AI-Module) — Odoo 19 Enterprise AI integration
- [Prompts](https://github.com/AlanOgic/odoo-mcp-19/wiki/Prompts) — the 19 guided workflow prompts

**Operations**

- [Deployment](https://github.com/AlanOgic/odoo-mcp-19/wiki/Deployment) — production HTTP behind Nginx, multi-client routing
- [MCP 2026-07-28 migration](docs/mcp-2026-07-28-migration.md) — why FastMCP is pinned below 4.x, and what moving costs

## Security

- **HTTP transport requires `MCP_API_KEY`** — server refuses to start without it
- **Docker runs as non-root user** (UID 1001)
- **Input validation** — model names (dotted notation regex), method names (identifier regex), URI scheme (`odoo://` only)
- **No traceback forwarding** — Odoo server tracebacks logged to stderr, never returned to clients
- **Credential files gitignored** — `.env`, `.mcp.json`, `odoo_config.json`
- **SSL warning** — visible warning on startup when `ODOO_VERIFY_SSL=false` with HTTPS
- **Thread-safe caches** — doc cache (100 entries max, LRU eviction) and runtime issue tracker use locks
- **`MCP_DEFAULT_CONTEXT`** — capped at 4KB, `MCP_BOOTSTRAP_MODELS` capped at 20 models

## Requirements

- Python 3.10+
- Odoo 19+
- FastMCP >=3.4.6,<4 (with tasks extra)
- requests 2.32.4+

## License

MIT
