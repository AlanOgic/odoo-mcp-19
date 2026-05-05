"""Tests for MCP_READ_ONLY enforcement in execute_method."""
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_ctx():
    return MagicMock()


def _import_execute_method():
    from odoo_mcp.server import execute_method
    return execute_method


def test_read_only_blocks_write(monkeypatch, mock_ctx):
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    execute_method = _import_execute_method()

    response = execute_method(
        ctx=mock_ctx,
        model="res.partner",
        method="write",
        args_json='[[1], {"name": "X"}]',
    )

    assert response.success is False
    assert "read-only" in response.error.lower()
    assert "MCP_READ_ONLY" in response.error


def test_read_only_blocks_create(monkeypatch, mock_ctx):
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    execute_method = _import_execute_method()

    response = execute_method(
        ctx=mock_ctx,
        model="res.partner",
        method="create",
        args_json='[{"name": "Test"}]',
    )

    assert response.success is False
    assert "read-only" in response.error.lower()


def test_read_only_blocks_action_method(monkeypatch, mock_ctx):
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    execute_method = _import_execute_method()

    response = execute_method(
        ctx=mock_ctx,
        model="sale.order",
        method="action_confirm",
        args_json='[[1]]',
    )

    assert response.success is False
    assert "read-only" in response.error.lower()


def test_read_only_allows_safe_method(monkeypatch, mock_ctx):
    """Read-only must NOT short-circuit safe reads — they should reach the
    Odoo client. We can't easily run a real call here, but we can check that
    the read-only guard does not produce a 'read-only mode is active' error
    for a safe method even when MCP_READ_ONLY=true."""
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    execute_method = _import_execute_method()

    response = execute_method(
        ctx=mock_ctx,
        model="not a model",   # Will fail _validate_model regex
        method="search_read",
    )

    # The error should be about model validation, NOT read-only mode.
    assert response.success is False
    assert "read-only" not in response.error.lower()


def test_read_only_off_does_not_block(monkeypatch, mock_ctx):
    """When MCP_READ_ONLY is unset/false, the guard must not fire even for
    write methods. We trigger model validation failure to short-circuit
    before the actual Odoo call."""
    monkeypatch.delenv("MCP_READ_ONLY", raising=False)
    execute_method = _import_execute_method()

    response = execute_method(
        ctx=mock_ctx,
        model="not a model",
        method="write",
    )

    assert response.success is False
    assert "read-only" not in response.error.lower()
