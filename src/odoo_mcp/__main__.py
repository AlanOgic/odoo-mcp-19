"""
Entry point for running the Odoo MCP server
"""

from .server import mcp


def main():
    """Main entry point for the MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
