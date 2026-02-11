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
  -e ODOO_URL="${ODOO_URL}" \
  -e ODOO_DB="${ODOO_DB}" \
  -e ODOO_USERNAME="${ODOO_USERNAME}" \
  -e ODOO_API_KEY="${ODOO_API_KEY}" \
  -e ODOO_PASSWORD="${ODOO_PASSWORD:-}" \
  -e ODOO_TIMEOUT="${ODOO_TIMEOUT:-30}" \
  -e ODOO_VERIFY_SSL="${ODOO_VERIFY_SSL:-true}" \
  odoo-mcp-19:doc-bearer
