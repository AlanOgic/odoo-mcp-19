"""Tests for MCP_READ_ONLY enforcement in execute_method."""
import asyncio
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


def test_read_only_blocks_batch_with_writes(monkeypatch):
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    from odoo_mcp.server import batch_execute

    response = asyncio.run(batch_execute(
        operations=[
            {"model": "res.partner", "method": "write", "args_json": '[[1], {"name": "X"}]'},
        ],
    ))

    assert response.success is False
    assert "read-only" in (response.error or "").lower()


def test_read_only_allows_batch_of_reads(monkeypatch):
    """A batch where every operation is SAFE should not be rejected by the
    read-only guard — the read-only guard must not return an error for it.

    Note: batch_execute uses FastMCP Progress which requires MCP dependency injection.
    When called directly (outside MCP context), it raises AssertionError after the
    read-only guard passes. We verify the read-only guard did NOT fire by confirming
    we get through it (either a response without 'read-only' or the Progress exception).
    """
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    from odoo_mcp.server import batch_execute

    try:
        response = asyncio.run(batch_execute(
            operations=[
                {"model": "not a model", "method": "search_read"},
            ],
        ))
        # If we got a response, the error must not be about read-only.
        assert "read-only" not in (response.error or "").lower()
    except AssertionError as exc:
        # FastMCP Progress requires MCP dependency injection outside MCP context.
        # Reaching this point means the read-only guard did not fire (correct).
        assert "read-only" not in str(exc).lower()


def test_read_only_blocks_workflow(monkeypatch):
    """Under read-only, ALL workflows are rejected (they're inherently multi-step)."""
    monkeypatch.setenv("MCP_READ_ONLY", "true")
    from odoo_mcp.server import execute_workflow

    response = asyncio.run(execute_workflow(
        workflow="quote_to_cash",
        params_json='{"partner_id": 1, "product_id": 1, "quantity": 1}',
    ))

    assert response.success is False
    assert "read-only" in (response.error or "").lower()


def test_read_only_off_allows_workflow_to_proceed_to_validation(monkeypatch):
    """When read-only is off, the workflow guard does not fire. (We don't actually
    expect a successful run — just absence of the read-only error.)"""
    monkeypatch.delenv("MCP_READ_ONLY", raising=False)
    from odoo_mcp.server import execute_workflow

    response = asyncio.run(execute_workflow(
        workflow="quote_to_cash",
        params_json='{"partner_id": 1, "product_id": 1, "quantity": 1}',
    ))

    # Whatever happens downstream, the error (if any) must not be about read-only.
    assert "read-only" not in (response.error or "").lower()
