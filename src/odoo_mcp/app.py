"""
FastMCP application setup for the Odoo MCP Server.

Creates the FastMCP instance with auth, lifespan, and icon loading.
Other modules import `mcp` from here to register resources, tools, and prompts.
"""

import base64
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

from fastmcp import FastMCP
from fastmcp_tasks import TasksExtension
from mcp.types import Icon

from .odoo_client import OdooClient, get_env_client

logger = logging.getLogger(__name__)


# ----- Icon Loading -----


def _load_icon() -> Optional[Icon]:
    """Load the Odoo icon from assets as a data URI."""
    icon_path = Path(__file__).parent / "assets" / "odoo_icon.svg"
    try:
        if icon_path.exists():
            icon_data = base64.standard_b64encode(icon_path.read_bytes()).decode()
            return Icon(
                src=f"data:image/svg+xml;base64,{icon_data}",
                mime_type="image/svg+xml",
            )
    except Exception as e:
        logger.warning("Could not load icon: %s", e)
    return None


ODOO_ICON = _load_icon()


# ----- Application Lifespan -----


@dataclass
class AppContext:
    """Application context for the MCP server"""

    odoo: Optional[OdooClient]


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Application lifespan for initialization and cleanup.

    In multi-user mode (USERS_DB_PATH) the env Odoo credentials are
    optional — per-user clients are built lazily from the registry, so a
    missing env config must not prevent startup.
    """
    try:
        odoo_client: Optional[OdooClient] = get_env_client()
    except (FileNotFoundError, KeyError) as e:
        if os.environ.get("USERS_DB_PATH"):
            logger.warning("No env Odoo credentials (%s) — multi-user registry mode only", e)
            odoo_client = None
        else:
            raise
    try:
        yield AppContext(odoo=odoo_client)
    finally:
        pass


# ----- Authentication -----


def _get_auth_provider():
    """Get the auth provider.

    USERS_DB_PATH set -> per-user keys from the shared registry, with the
    static MCP_API_KEY (if any) kept as fallback. Otherwise today's
    single-static-key behavior, or no auth (stdio).
    """
    api_key = os.environ.get("MCP_API_KEY")

    from .users_db import get_users_db

    users_db = get_users_db()
    if users_db is not None:
        from .auth_verifier import DbTokenVerifier
        from .user_clients import declare_registry_auth

        declare_registry_auth(users_db)
        return DbTokenVerifier(users_db, static_api_key=api_key)

    if api_key:
        from fastmcp.server.auth import StaticTokenVerifier

        return StaticTokenVerifier(
            tokens={
                api_key: {
                    "client_id": "mcp-client",
                    "scopes": ["read", "write"],
                }
            }
        )
    return None


# ----- Create MCP Server -----

_auth = _get_auth_provider()
_icons = [ODOO_ICON] if ODOO_ICON else None

mcp = FastMCP(
    "Odoo 19+ MCP Server",
    lifespan=app_lifespan,
    auth=_auth,
    website_url="https://github.com/AlanOgic/odoo-mcp-19",
    icons=_icons,
)

# batch_execute and execute_workflow are task=True tools. Since FastMCP 4 background
# tasks are an extension (io.modelcontextprotocol/tasks): without it the server refuses
# to start. Registered here, before server.py declares the tools.
mcp.add_extension(TasksExtension())
