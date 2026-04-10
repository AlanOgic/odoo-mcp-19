#!/bin/bash
# Wrapper script for Claude Desktop to run MCP server in Docker

# Load .env file if it exists (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

docker run --rm -i \
  --env-file "$SCRIPT_DIR/.env" \
  odoo-mcp-19:latest
