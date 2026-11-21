"""A setting given in the process environment must win over a ``.env`` file.

``load_config()`` runs after startup (lifespan) and used ``override=True``, so a
``MCP_READ_ONLY=false`` line in ``.env`` undid ``MCP_READ_ONLY=true`` given on the
command line or by the container.
"""

import os
from unittest.mock import patch

import pytest

from odoo_mcp.odoo_client import load_config
from odoo_mcp.safety_profile import get_profile

ENV_FILE = """\
ODOO_URL=https://from-file.example
ODOO_DB=file_db
ODOO_USERNAME=file_user
ODOO_API_KEY=file_key
MCP_READ_ONLY=false
"""


@pytest.fixture
def env_file_dir(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(ENV_FILE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ODOO_CONFIG_DIR", raising=False)
    return tmp_path


def _environment(**values):
    """os.environ without any ODOO_* / MCP_READ_ONLY, plus ``values``; restored on exit."""
    kept = {k: v for k, v in os.environ.items() if not k.startswith("ODOO_") and k != "MCP_READ_ONLY"}
    return patch.dict(os.environ, {**kept, **values}, clear=True)


def test_the_process_environment_wins_over_the_env_file(env_file_dir):
    with _environment(MCP_READ_ONLY="true", ODOO_URL="https://from-env.example"):
        config = load_config()

        assert os.environ["MCP_READ_ONLY"] == "true"
        assert get_profile().read_only is True
        assert config["url"] == "https://from-env.example"


def test_the_env_file_fills_what_the_environment_does_not_set(env_file_dir):
    with _environment():
        config = load_config()

        assert config == {
            "url": "https://from-file.example",
            "db": "file_db",
            "username": "file_user",
            "password": "",
        }
        assert os.environ["MCP_READ_ONLY"] == "false"
