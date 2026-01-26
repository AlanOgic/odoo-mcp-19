#!/bin/bash
# Wrapper script for Claude Desktop to run MCP server in Docker

docker run --rm -i \
  -e ODOO_URL="${ODOO_URL}" \
  -e ODOO_DB="${ODOO_DB}" \
  -e ODOO_USERNAME="${ODOO_USERNAME}" \
  -e ODOO_API_KEY="${ODOO_API_KEY}" \
  -e ODOO_PASSWORD="${ODOO_PASSWORD:-}" \
  -e ODOO_TIMEOUT="${ODOO_TIMEOUT:-30}" \
  -e ODOO_VERIFY_SSL="${ODOO_VERIFY_SSL:-true}" \
  odoo-mcp-19
