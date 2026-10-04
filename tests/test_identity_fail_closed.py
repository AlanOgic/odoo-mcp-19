"""In multi-user mode a call without the caller's identity must never run as env-admin.

The caller's access token lives in contextvars. Two ways to lose it were found:
``concurrent.futures.ThreadPoolExecutor`` does not copy contextvars (unlike
``anyio.to_thread``), so the bundle and session-bootstrap workers saw no token;
and any lost context fell through ``get_odoo_client()`` to the env singleton with
role ``None``, which the safety layer treats as the privileged operator.

Only a server the entry point started on STDIO, where nobody authenticates, may
treat a token-less call as the operator; the decision is taken once at startup so
no environment variable can switch it off later.

These tests drive the real auth path: a registry key goes through the real
``DbTokenVerifier`` and the resulting ``AccessToken`` is installed in the same
contextvar FastMCP's bearer middleware sets. Only the Odoo HTTP call is faked.
"""

import asyncio
import json
import sqlite3
from datetime import datetime

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser

import odoo_mcp.odoo_client as odoo_client_module
import odoo_mcp.user_clients as user_clients
import odoo_mcp.users_db as users_db_module
from odoo_mcp import __main__ as entry_point
from odoo_mcp import app, resources
from odoo_mcp.auth_verifier import ENV_ADMIN_CLIENT_ID, DbTokenVerifier
from odoo_mcp.users_db import UsersDb
from tests.conftest import TEST_ENCRYPTION_KEY, encrypt_with_contract

MEMBER_ODOO_USERNAME = "thierry@cyanview.com"  # stored for "member" by users_db_seed
ENV_ADMIN_USERNAME = "env-admin@example.com"
FIELDS = {"name": {"type": "char", "string": "Name", "required": True}}


@pytest.fixture
def odoo_calls(monkeypatch):
    """Record which Odoo account each fields_get is sent as; never touch the network."""
    calls: list[tuple[str, str]] = []

    def fake_execute(self, model, method, *args, **kwargs):
        calls.append((self.username, model))
        return FIELDS

    monkeypatch.setattr(odoo_client_module.OdooClient, "_execute", fake_execute)
    return calls


@pytest.fixture(autouse=True)
def _registry_mode(monkeypatch, users_db_seed):
    """A registry is configured and the process was not started on STDIO."""
    monkeypatch.setenv("USERS_DB_PATH", str(users_db_seed.db_path))
    monkeypatch.setenv("ODOO_URL", "https://odoo.example.com")
    monkeypatch.setenv("ODOO_DB", "cyanview")
    monkeypatch.setattr(users_db_module, "_users_db", None)
    monkeypatch.setattr(user_clients, "_cache", {})
    monkeypatch.setattr(user_clients, "_serving_stdio", False)
    monkeypatch.setattr(user_clients, "_registry", None)
    # The env singleton, built without load_config() so no real .env is read.
    env_client = odoo_client_module.OdooClient(
        url="https://odoo.example.com", db="cyanview", username=ENV_ADMIN_USERNAME, api_key="env-key"
    )
    monkeypatch.setattr(odoo_client_module, "get_env_client", lambda: env_client)
    yield


def _authenticate(db_path, key: str, static_api_key: str | None = None):
    verifier = DbTokenVerifier(UsersDb(db_path), static_api_key=static_api_key)
    token = asyncio.run(verifier.verify_token(key))
    assert token is not None
    return token


@pytest.fixture
def signed_in():
    """Install an AccessToken exactly where FastMCP's bearer middleware puts it."""
    resets = []

    def _sign_in(token):
        resets.append(auth_context_var.set(AuthenticatedUser(token)))
        return token

    yield _sign_in
    for reset in reversed(resets):
        auth_context_var.reset(reset)


@pytest.fixture
def as_member(users_db_seed, signed_in):
    return signed_in(_authenticate(users_db_seed.db_path, users_db_seed.keys["member_odoo"]))


class TestParallelSchemaFetchesKeepTheCaller:
    def test_bundle_reads_every_schema_as_the_calling_user(self, as_member, odoo_calls):
        payload = json.loads(resources.get_bundle("res.partner,sale.order"))

        assert payload["errors"] == {}
        assert sorted(odoo_calls) == [(MEMBER_ODOO_USERNAME, "res.partner"), (MEMBER_ODOO_USERNAME, "sale.order")]

    def test_session_bootstrap_reads_every_schema_as_the_calling_user(self, monkeypatch, as_member, odoo_calls):
        monkeypatch.setenv("MCP_BOOTSTRAP_MODELS", "res.partner,sale.order")

        payload = json.loads(resources.get_session_bootstrap())

        assert payload["errors"] == {}
        assert sorted(odoo_calls) == [(MEMBER_ODOO_USERNAME, "res.partner"), (MEMBER_ODOO_USERNAME, "sale.order")]


class TestParallelMapHelper:
    def test_results_come_back_in_input_order(self):
        assert resources._map_in_caller_context(str.upper, ["c", "a", "b"]) == ["C", "A", "B"]

    def test_an_exception_in_one_call_propagates(self):
        def fail_on_b(item: str) -> str:
            if item == "b":
                raise PermissionError("lost caller")
            return item

        with pytest.raises(PermissionError, match="lost caller"):
            resources._map_in_caller_context(fail_on_b, ["a", "b"])

    def test_no_items_means_no_pool_and_no_results(self):
        assert resources._map_in_caller_context(str.upper, []) == []


class TestMissingIdentityIsRefused:
    def test_no_odoo_client_without_a_caller(self):
        with pytest.raises(PermissionError, match="authenticated"):
            odoo_client_module.get_odoo_client()

    def test_no_role_without_a_caller(self):
        """Role None is the privileged operator: a lost context must not become it."""
        with pytest.raises(PermissionError, match="authenticated"):
            user_clients.current_role()

    def test_mcp_transport_in_the_environment_cannot_switch_the_check_off(self, monkeypatch):
        """A server started without the STDIO declaration (fastmcp run, an ASGI
        server on mcp.http_app(), or a .env reloaded after startup) stays closed."""
        monkeypatch.setenv("MCP_TRANSPORT", "stdio")

        with pytest.raises(PermissionError, match="authenticated"):
            odoo_client_module.get_odoo_client()

    def test_a_lost_caller_in_bundle_is_refused_not_served_as_env_admin(self, odoo_calls):
        with pytest.raises(PermissionError, match="authenticated"):
            resources.get_bundle("res.partner")

        assert odoo_calls == []

    def test_a_lost_caller_in_session_bootstrap_is_refused_not_reported_per_model(self, monkeypatch, odoo_calls):
        monkeypatch.setenv("MCP_BOOTSTRAP_MODELS", "res.partner")

        with pytest.raises(PermissionError, match="authenticated"):
            resources.get_session_bootstrap()

        assert odoo_calls == []


class TestTheEntryPointDeclaresStdio:
    @pytest.fixture(autouse=True)
    def runs(self, monkeypatch):
        monkeypatch.setenv("MCP_VERBOSE", "false")
        calls: list[dict] = []
        monkeypatch.setattr(entry_point.mcp, "run", lambda **kwargs: calls.append(kwargs))
        return calls

    def test_a_stdio_run_asks_fastmcp_for_stdio_explicitly(self, monkeypatch, runs):
        """A bare mcp.run() takes FASTMCP_TRANSPORT (env or .env): it could serve HTTP
        while the process has declared STDIO."""
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        monkeypatch.setenv("FASTMCP_TRANSPORT", "http")

        entry_point.main()

        assert [call.get("transport") for call in runs] == ["stdio"]

    def test_a_stdio_run_serves_token_less_calls_as_the_operator(self, monkeypatch):
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)

        entry_point.main()

        assert odoo_client_module.get_odoo_client().username == ENV_ADMIN_USERNAME
        assert user_clients.current_role() is None

    def test_an_http_run_keeps_token_less_calls_refused(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
        monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", TEST_ENCRYPTION_KEY)

        entry_point.main()

        with pytest.raises(PermissionError, match="authenticated"):
            odoo_client_module.get_odoo_client()


class TestTheRegistryIsFixedAtStartup:
    """``load_config()`` reloads .env files at runtime: a ``USERS_DB_PATH=`` line there
    must neither switch the check off nor send registry users to the env account."""

    @pytest.fixture
    def registry_installed_then_emptied(self, monkeypatch):
        assert isinstance(app._get_auth_provider(), DbTokenVerifier)
        monkeypatch.setenv("USERS_DB_PATH", "")
        monkeypatch.setattr(users_db_module, "_users_db", None)

    def test_token_less_calls_stay_refused(self, registry_installed_then_emptied):
        with pytest.raises(PermissionError, match="authenticated"):
            odoo_client_module.get_odoo_client()

    def test_registry_users_keep_their_own_client(self, registry_installed_then_emptied, as_member):
        assert odoo_client_module.get_odoo_client().username == MEMBER_ODOO_USERNAME


class TestIdentitiesThatUseTheEnvClient:
    def test_the_static_api_key_identity_uses_the_env_client(self, users_db_seed, signed_in):
        signed_in(_authenticate(users_db_seed.db_path, "static-admin-key", static_api_key="static-admin-key"))

        assert odoo_client_module.get_odoo_client().username == ENV_ADMIN_USERNAME
        assert user_clients.current_role() == "admin"

    def test_a_registry_user_gets_their_own_client(self, as_member):
        assert odoo_client_module.get_odoo_client().username == MEMBER_ODOO_USERNAME
        assert user_clients.current_role() == "support"

    def test_a_registry_user_whose_id_reads_env_admin_still_gets_their_own_client(
        self, users_db_seed, signed_in, odoo_calls
    ):
        """The env account belongs to the static key, not to a client_id string."""
        now = datetime.now().isoformat()
        conn = sqlite3.connect(users_db_seed.db_path)
        conn.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ENV_ADMIN_CLIENT_ID, "Lookalike", "lookalike@cyanview.com", "support", 1, now, now),
        )
        users_db_seed.add_key(conn, "lookalike", ENV_ADMIN_CLIENT_ID, "odoo")
        conn.execute(
            "INSERT INTO user_odoo_credentials VALUES (?, ?, ?, ?)",
            (
                ENV_ADMIN_CLIENT_ID,
                "lookalike@cyanview.com",
                encrypt_with_contract({"api_key": "lookalike-key"}, users_db_seed.salt, TEST_ENCRYPTION_KEY),
                now,
            ),
        )
        conn.commit()
        conn.close()

        signed_in(_authenticate(users_db_seed.db_path, users_db_seed.keys["lookalike"]))

        assert odoo_client_module.get_odoo_client().username == "lookalike@cyanview.com"
