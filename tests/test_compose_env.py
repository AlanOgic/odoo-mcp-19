"""docker-compose.yml must forward every variable the safety profile reads.

Compose uses ``.env`` for interpolation only: a variable the ``environment:`` list
does not name never reaches the container, so ``MCP_READ_ONLY=true`` in ``.env``
used to leave writes enabled.
"""

import re
from pathlib import Path

import pytest

COMPOSE = (Path(__file__).resolve().parent.parent / "docker-compose.yml").read_text()

OVERRIDES = ["MCP_READ_ONLY", "MCP_WRITE_ALLOWLIST", "MCP_VALIDATE_PAYLOADS"]


@pytest.mark.parametrize("name", OVERRIDES)
def test_safety_override_is_passed_through_without_a_default(name):
    """Bare name = passed through when set, absent otherwise.

    ``${VAR:-}`` would hand the container an empty string, which the profile reads
    as ``false`` and which would switch off the ``locked`` defaults.
    """
    assert re.search(rf"^\s*- {name}\s*$", COMPOSE, re.MULTILINE), f"{name} is not forwarded"


def test_safety_mode_is_forwarded():
    assert "MCP_SAFETY_MODE=" in COMPOSE
