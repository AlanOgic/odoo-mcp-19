"""The server ships generic Odoo prompts only.

The Cyanview workflow skills (``cyanview-*``) were served here as prompts until
v2.1.0. They now live exclusively in the private skills repository, installed
client side, so this server must not publish them again, ship their bodies, or
keep the per-user filter that only existed for them.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

import odoo_mcp
import odoo_mcp.server  # noqa: F401  registers every prompt the server publishes
from odoo_mcp.app import mcp

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    tomllib = pytest.importorskip("tomli", reason="3.10 needs tomli to read pyproject")

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_PROMPT_PREFIX = "cyanview-"


def test_no_skill_prompt_is_published():
    names = {p.name for p in asyncio.run(mcp.list_prompts())}
    assert names, "no prompt registered: the assertion below would pass vacuously"
    assert {n for n in names if n.startswith(SKILL_PROMPT_PREFIX)} == set()


def test_skill_bodies_are_not_packaged():
    assert not (Path(odoo_mcp.__file__).parent / "skills").exists()
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]["odoo_mcp"]
    assert not [glob for glob in package_data if glob.startswith("skills/")]


@pytest.mark.parametrize("module", ["odoo_mcp.skill_prompts", "odoo_mcp.skill_visibility"])
def test_skill_prompt_modules_are_gone(module):
    """The per-user filter only existed for the skill prompts; multi-user mode must not
    register it (it is added at import time, only when USERS_DB_PATH is set)."""
    assert importlib.util.find_spec(module) is None
