"""Live integration test for MCP_SAFETY_MODE=locked.

Mutates env state. Run directly with python:
    python tests/live/test_locked_mode_live.py

Requires .env in cwd with valid Odoo credentials.
"""
import json
import os
import sys
from pathlib import Path

# Make src/ importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def _reset_env():
    for key in (
        "MCP_SAFETY_MODE", "MCP_READ_ONLY", "MCP_WRITE_ALLOWLIST",
        "MCP_HOST", "MCP_VALIDATE_PAYLOADS",
    ):
        os.environ.pop(key, None)


def test_locked_blocks_write():
    _reset_env()
    os.environ["MCP_SAFETY_MODE"] = "locked"
    from odoo_mcp.server import execute_method
    from unittest.mock import MagicMock

    response = execute_method(
        ctx=MagicMock(),
        model="res.partner",
        method="write",
        args_json='[[1], {"name": "live-test"}]',
    )
    assert not response.success
    assert "read-only" in (response.error or "").lower()
    print("✓ locked blocks write")


def test_locked_allows_safe_read():
    _reset_env()
    os.environ["MCP_SAFETY_MODE"] = "locked"
    from odoo_mcp.server import execute_method
    from unittest.mock import MagicMock

    response = execute_method(
        ctx=MagicMock(),
        model="res.partner",
        method="search_read",
        args_json='[[]]',
        kwargs_json='{"fields": ["name"], "limit": 1}',
    )
    assert response.success, f"locked should permit reads, got: {response.error}"
    print("✓ locked allows safe read")


def test_allowlist_unblocks_named_method():
    _reset_env()
    os.environ["MCP_SAFETY_MODE"] = "locked"
    os.environ["MCP_READ_ONLY"] = "false"  # Allow writes generally...
    os.environ["MCP_WRITE_ALLOWLIST"] = "res.partner.message_post"
    from odoo_mcp.safety import classify_operation, RiskLevel

    # ...but only message_post on res.partner is allowed.
    permitted = classify_operation(
        "res.partner", "message_post", args=[[1], "live test"],
    )
    blocked = classify_operation(
        "res.partner", "write", args=[[1], {"name": "X"}],
    )
    assert permitted.risk_level is not RiskLevel.BLOCKED
    assert blocked.risk_level is RiskLevel.BLOCKED
    print("✓ allowlist unblocks named method, blocks others")


def test_server_status_resource():
    _reset_env()
    os.environ["MCP_SAFETY_MODE"] = "locked"
    from odoo_mcp.server import read_resource

    raw = read_resource("odoo://server-status")
    payload = json.loads(raw)
    assert payload["safety_mode"] == "locked"
    assert payload["read_only"] is True
    assert payload["host"] == "127.0.0.1"
    print("✓ server-status reports locked posture")


def test_payload_validation_catches_bad_field():
    _reset_env()
    os.environ["MCP_SAFETY_MODE"] = "strict"  # so writes are allowed
    os.environ["MCP_VALIDATE_PAYLOADS"] = "true"
    from odoo_mcp.server import execute_method
    from unittest.mock import MagicMock

    response = execute_method(
        ctx=MagicMock(),
        model="res.partner",
        method="write",
        args_json='[[1], {"definitely_not_a_field": "x"}]',
    )
    assert not response.success
    assert "definitely_not_a_field" in (response.error or "")
    print("✓ payload pre-flight catches hallucinated field")


if __name__ == "__main__":
    test_locked_blocks_write()
    test_locked_allows_safe_read()
    test_allowlist_unblocks_named_method()
    test_server_status_resource()
    test_payload_validation_catches_bad_field()
    print("\nAll locked-mode live checks passed.")
