"""Tests for odoo://server-status resource."""
import json


def _read_status() -> dict:
    """Read the server-status resource via the read_resource bridge."""
    from odoo_mcp.server import read_resource
    raw = read_resource("odoo://server-status")
    return json.loads(raw)


def test_status_includes_version(monkeypatch):
    monkeypatch.delenv("MCP_SAFETY_MODE", raising=False)
    status = _read_status()
    assert "version" in status
    assert isinstance(status["version"], str)


def test_status_reflects_locked_mode(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "locked")
    monkeypatch.delenv("MCP_HOST", raising=False)
    monkeypatch.delenv("MCP_READ_ONLY", raising=False)
    status = _read_status()
    assert status["safety_mode"] == "locked"
    assert status["read_only"] is True
    assert status["host"] == "127.0.0.1"
    assert status["validate_payloads"] is True


def test_status_under_strict(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "strict")
    monkeypatch.delenv("MCP_HOST", raising=False)
    monkeypatch.delenv("MCP_READ_ONLY", raising=False)
    monkeypatch.delenv("MCP_WRITE_ALLOWLIST", raising=False)
    monkeypatch.delenv("MCP_VALIDATE_PAYLOADS", raising=False)

    status = _read_status()
    assert status["safety_mode"] == "strict"
    assert status["read_only"] is False
    assert status["validate_payloads"] is False
    assert status["host"] == "0.0.0.0"
    assert status["write_allowlist"] == []


def test_status_does_not_leak_secrets(monkeypatch):
    monkeypatch.setenv("MCP_API_KEY", "supersecret-token-12345")
    monkeypatch.setenv("ODOO_API_KEY", "odoo-key-xyz")
    monkeypatch.setenv("ODOO_PASSWORD", "hunter2")

    status = _read_status()
    raw = json.dumps(status)

    assert "supersecret" not in raw
    assert "odoo-key-xyz" not in raw
    assert "hunter2" not in raw


def test_status_includes_warnings_array(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "locked")
    monkeypatch.setenv("MCP_HOST", "0.0.0.0")  # foot-gun: locked + remote
    status = _read_status()
    assert isinstance(status["warnings"], list)
    assert len(status["warnings"]) >= 1
    assert any("0.0.0.0" in w for w in status["warnings"])


def test_status_includes_posture_open(monkeypatch):
    monkeypatch.setenv("MCP_SAFETY_MODE", "permissive")
    monkeypatch.setenv("MCP_HOST", "0.0.0.0")
    monkeypatch.delenv("MCP_WRITE_ALLOWLIST", raising=False)
    monkeypatch.delenv("MCP_READ_ONLY", raising=False)

    status = _read_status()
    assert status["posture_open"] is True
