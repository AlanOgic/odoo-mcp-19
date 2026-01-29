"""
Entry point for running the Odoo MCP server
"""

import os

from .server import mcp


def main():
    """Main entry point for the MCP server"""
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8080"))

    # Log auth status for HTTP transport
    if transport == "streamable-http":
        if os.environ.get("MCP_API_KEY"):
            print("🔐 Authentication enabled (Bearer token required)")
        else:
            print("⚠️  No MCP_API_KEY set - running without authentication")
        mcp.run(transport="streamable-http", host=host, port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
