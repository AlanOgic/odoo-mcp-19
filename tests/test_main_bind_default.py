"""Tests for MCP_HOST default-resolution flip under locked mode."""

from odoo_mcp.safety_profile import resolve


def test_strict_mode_default_host_is_0_0_0_0():
    profile = resolve({"MCP_SAFETY_MODE": "strict"})
    assert profile.host == "0.0.0.0"


def test_locked_mode_default_host_is_localhost():
    profile = resolve({"MCP_SAFETY_MODE": "locked"})
    assert profile.host == "127.0.0.1"


def test_explicit_host_overrides_locked_default():
    profile = resolve({"MCP_SAFETY_MODE": "locked", "MCP_HOST": "0.0.0.0"})
    assert profile.host == "0.0.0.0"


def test_main_module_imports_cleanly(monkeypatch):
    """Smoke check — importing __main__ should not error after the host
    resolution change."""
    import importlib
    from odoo_mcp import __main__ as main_mod
    importlib.reload(main_mod)
    assert hasattr(main_mod, "main")
