"""Tests for safety profile resolver."""
import pytest

from odoo_mcp.safety_profile import (
    ResolvedProfile,
    SafetyMode,
    resolve,
)


def test_default_mode_is_strict():
    profile = resolve({})
    assert profile.safety_mode is SafetyMode.STRICT
    assert profile.read_only is False
    assert profile.write_allowlist == frozenset()
    assert profile.write_allowlist_enforced is False
    assert profile.host_default == "0.0.0.0"
    assert profile.validate_payloads is False


def test_permissive_mode():
    profile = resolve({"MCP_SAFETY_MODE": "permissive"})
    assert profile.safety_mode is SafetyMode.PERMISSIVE
    assert profile.read_only is False
    assert profile.write_allowlist_enforced is False
    assert profile.host_default == "0.0.0.0"
    assert profile.validate_payloads is False


def test_locked_mode_defaults():
    profile = resolve({"MCP_SAFETY_MODE": "locked"})
    assert profile.safety_mode is SafetyMode.LOCKED
    assert profile.read_only is True
    assert profile.write_allowlist_enforced is True
    assert profile.host_default == "127.0.0.1"
    assert profile.validate_payloads is True


def test_unknown_mode_falls_back_to_strict():
    profile = resolve({"MCP_SAFETY_MODE": "garbage"})
    assert profile.safety_mode is SafetyMode.STRICT


def test_read_only_override_beats_strict_default():
    profile = resolve({"MCP_SAFETY_MODE": "strict", "MCP_READ_ONLY": "true"})
    assert profile.safety_mode is SafetyMode.STRICT
    assert profile.read_only is True


def test_read_only_override_beats_locked_default():
    profile = resolve({"MCP_SAFETY_MODE": "locked", "MCP_READ_ONLY": "false"})
    assert profile.safety_mode is SafetyMode.LOCKED
    assert profile.read_only is False


def test_allowlist_explicit_under_strict_enables_enforcement():
    profile = resolve({
        "MCP_SAFETY_MODE": "strict",
        "MCP_WRITE_ALLOWLIST": "sale.order.action_confirm,res.partner.message_post",
    })
    assert profile.write_allowlist_enforced is True
    assert "sale.order.action_confirm" in profile.write_allowlist
    assert "res.partner.message_post" in profile.write_allowlist


def test_allowlist_wildcard_normalised():
    profile = resolve({"MCP_WRITE_ALLOWLIST": "sale.order.*,product.product.write"})
    assert "sale.order.*" in profile.write_allowlist
    assert "product.product.write" in profile.write_allowlist


def test_allowlist_whitespace_tolerant():
    profile = resolve({
        "MCP_WRITE_ALLOWLIST": "  sale.order.action_confirm , product.product.write ",
    })
    assert profile.write_allowlist == frozenset({
        "sale.order.action_confirm",
        "product.product.write",
    })


def test_host_default_under_locked_is_localhost():
    profile = resolve({"MCP_SAFETY_MODE": "locked"})
    assert profile.host_default == "127.0.0.1"


def test_host_explicit_overrides_locked_default():
    profile = resolve({"MCP_SAFETY_MODE": "locked", "MCP_HOST": "0.0.0.0"})
    assert profile.host == "0.0.0.0"


def test_host_unset_resolves_to_default_per_mode():
    locked = resolve({"MCP_SAFETY_MODE": "locked"})
    assert locked.host == "127.0.0.1"
    strict = resolve({"MCP_SAFETY_MODE": "strict"})
    assert strict.host == "0.0.0.0"


def test_validate_payloads_explicit_enable_under_strict():
    profile = resolve({"MCP_SAFETY_MODE": "strict", "MCP_VALIDATE_PAYLOADS": "true"})
    assert profile.validate_payloads is True


def test_posture_open_only_when_loose():
    open_ = resolve({"MCP_SAFETY_MODE": "permissive", "MCP_HOST": "0.0.0.0"})
    assert open_.posture_open is True

    locked = resolve({"MCP_SAFETY_MODE": "locked"})
    assert locked.posture_open is False

    permissive_with_allowlist = resolve({
        "MCP_SAFETY_MODE": "permissive",
        "MCP_HOST": "0.0.0.0",
        "MCP_WRITE_ALLOWLIST": "res.partner.message_post",
    })
    assert permissive_with_allowlist.posture_open is False


def test_warnings_flag_locked_with_remote_bind():
    profile = resolve({"MCP_SAFETY_MODE": "locked", "MCP_HOST": "0.0.0.0"})
    assert any("0.0.0.0" in w for w in profile.warnings)


def test_warnings_empty_under_default_strict():
    profile = resolve({})
    assert profile.warnings == ()


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ("", False), ("garbage", False),
])
def test_bool_env_parsing(raw: str, expected: bool):
    profile = resolve({"MCP_READ_ONLY": raw})
    assert profile.read_only is expected
