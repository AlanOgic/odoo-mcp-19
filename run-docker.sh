#!/bin/bash
# Wrapper script for Claude Desktop to run MCP server in Docker

# Load .env file if it exists (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

# Force STDIO: this wrapper is launched by MCP clients (Claude Desktop / Claude Code)
# that speak over stdin/stdout. .env may set MCP_TRANSPORT=streamable-http for
# docker compose; that would start an unreachable HTTP listener and hang the client.
docker run --rm -i \
  --env-file "$SCRIPT_DIR/.env" \
  -e MCP_TRANSPORT=stdio \
  odoo-mcp-19:latest
