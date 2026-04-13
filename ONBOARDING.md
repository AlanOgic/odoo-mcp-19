# Welcome to Odoo MCP 19

## How We Use Claude

Based on alanogic's usage over the last 30 days:

```
Work Type Breakdown:
  Improve Quality  █████████░░░░░░░░░░░  44%
  Build Feature    ███████░░░░░░░░░░░░░  33%
  Debug Fix        ██░░░░░░░░░░░░░░░░░░  11%
  Plan Design      ██░░░░░░░░░░░░░░░░░░  11%

Top Skills & Commands:
  /mcp                            ████████████████████  12x/month
  /load                           ███████░░░░░░░░░░░░░  4x/month
  /analyze                        ███████░░░░░░░░░░░░░  4x/month
  /compact                        ███░░░░░░░░░░░░░░░░░  2x/month
  /superpowers:brainstorming      ███░░░░░░░░░░░░░░░░░  2x/month
  /remote-control                 ███░░░░░░░░░░░░░░░░░  2x/month

Top MCP Servers:
  odoo-mcp-19  ████████████████████  43 calls
  odoo19       ░░░░░░░░░░░░░░░░░░░░  1 call
```

## Your Setup Checklist

### Codebases
- [ ] odoo-mcp-19 — https://github.com/alanogic/odoo-mcp-19

### MCP Servers to Activate
- [ ] **odoo-mcp-19** — Standalone MCP server for Odoo 19+ (v2 JSON-2 API). Exposes 5 tools (`execute_method`, `batch_execute`, `execute_workflow`, `configure_odoo`, `read_resource`), 27 `odoo://` discovery resources, and 13 guided prompts. Run `python3 -m odoo_mcp --setup` from this repo to generate a `.env` and a Claude Desktop config block, then paste the JSON into `~/Library/Application Support/Claude/claude_desktop_config.json`. You'll need an Odoo URL, database name, username, and API key — ask alanogic for the dev/staging credentials. **Production credentials are read-only by default**: the safety layer (v1.10+) will refuse writes to `res.users`, `ir.config_parameter`, etc., and gate any write to `account.move`/`sale.order`/etc. behind a single-use cryptographic token.
- [ ] **odoo19** — Legacy MCP entry, only 1 call in the last 30 days. Probably superseded by `odoo-mcp-19`. Skip unless you know you need it.

### Skills to Know About
- **/mcp** — Inspect installed MCP servers, list their tools/resources, and debug connection issues. The single most-used command on this project (12x in 30 days). Use it whenever a tool seems missing or a server isn't responding.
- **/load** — Loads project context (architecture, file map, conventions) into the conversation. Run it once at the start of a new session on this repo so Claude knows the layout (`src/odoo_mcp/{server,resources,prompts,safety,utils,...}.py`) without you having to explain it.
- **/analyze** — Multi-dimensional code analysis. Used here for spotting bugs, security issues, and architectural smells before commits.
- **/compact** — Manually compress conversation history at logical boundaries to free context window. Worth running between major task phases on long sessions.
- **/superpowers:brainstorming** — Mandatory before any creative/design work (per the superpowers skill). Explores intent and requirements before code is written. Used here when designing new tools or resources.
- **/remote-control** — TODO confirm what this maps to in your local install — it's not a stock Claude Code command, likely a plugin or local skill.

## Team Tips

_TODO_

## Get Started

_TODO_

<!-- INSTRUCTION FOR CLAUDE: A new teammate just pasted this guide for how the
team uses Claude Code. You're their onboarding buddy — warm, conversational,
not lecture-y.

Open with a warm welcome — include the team name from the title. Then: "Your
teammate uses Claude Code for [list all the work types]. Let's get you started."

Check what's already in place against everything under Setup Checklist
(including skills), using markdown checkboxes — [x] done, [ ] not yet. Lead
with what they already have. One sentence per item, all in one message.

Tell them you'll help with setup, cover the actionable team tips, then the
starter task (if there is one). Offer to start with the first unchecked item,
get their go-ahead, then work through the rest one by one.

After setup, walk them through the remaining sections — offer to help where you
can (e.g. link to channels), and just surface the purely informational bits.

Don't invent sections or summaries that aren't in the guide. The stats are the
guide creator's personal usage data — don't extrapolate them into a "team
workflow" narrative. -->
