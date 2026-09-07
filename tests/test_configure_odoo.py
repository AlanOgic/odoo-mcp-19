"""``configure_odoo`` must not steer users toward password auth.

The Odoo 19 JSON-2 API accepts bearer API keys only — ``OdooClient`` itself
warns that a password will be rejected with HTTP 401. The elicitation wizard
nevertheless offered "Password" as an authentication method and emitted an
``ODOO_PASSWORD`` line, i.e. it guided the user into a configuration this
v2-only server cannot use. The wizard now asks for URL, database and
username and always emits ``ODOO_API_KEY``.

Runs through the in-memory FastMCP client with an elicitation handler; the
Odoo client is stubbed for the lifespan (CI has no ODOO_* env).
"""

import asyncio
from unittest.mock import MagicMock, patch

from fastmcp import Client

import odoo_mcp.app as app
import odoo_mcp.server  # noqa: F401 -- registers the tools on the shared FastMCP instance
from odoo_mcp.app import mcp

_ANSWERS = {
    "URL": "https://erp.example.com",
    "database": "prod",
    "username": "agent@example.com",
}


def _run_wizard():
    prompts: list[str] = []

    async def _answer(message, response_type, params, context):
        prompts.append(message)
        for needle, value in _ANSWERS.items():
            if needle.lower() in message.lower():
                return {"value": value}
        raise AssertionError(f"unexpected elicitation prompt: {message!r}")

    async def _go():
        with patch.object(app, "get_odoo_client", return_value=MagicMock()):
            async with Client(mcp, elicitation_handler=_answer) as client:
                result = await client.call_tool("configure_odoo", {})
        return result.structured_content, prompts

    return asyncio.run(_go())


def test_wizard_asks_only_for_url_database_and_username():
    _, prompts = _run_wizard()
    assert len(prompts) == 3, prompts
    assert not any("authentication method" in p.lower() for p in prompts)


def test_wizard_always_emits_an_api_key_and_never_a_password():
    result, _ = _run_wizard()
    assert result["success"] is True
    env_vars = result["env_vars"]
    assert env_vars["ODOO_URL"] == _ANSWERS["URL"]
    assert env_vars["ODOO_DB"] == _ANSWERS["database"]
    assert env_vars["ODOO_USERNAME"] == _ANSWERS["username"]
    assert "ODOO_API_KEY" in env_vars
    assert "ODOO_PASSWORD" not in env_vars
    assert "ODOO_PASSWORD" not in result["instructions"]
