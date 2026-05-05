"""Tests for live fields_get pre-flight validation."""
from unittest.mock import MagicMock

import pytest

from odoo_mcp.safety import (
    validate_payload_against_schema,
)


def _client_with_fields(fields: dict):
    client = MagicMock()
    client.execute_kw.return_value = fields
    return client


@pytest.fixture(autouse=True)
def clear_fields_cache():
    """Each test must see fresh fields_get behaviour."""
    from odoo_mcp.utils import _FIELDS_CACHE, _FIELDS_CACHE_LOCK
    with _FIELDS_CACHE_LOCK:
        _FIELDS_CACHE.clear()
    yield
    with _FIELDS_CACHE_LOCK:
        _FIELDS_CACHE.clear()


def test_valid_payload_passes():
    client = _client_with_fields({
        "name": {"type": "char", "readonly": False, "required": True},
        "email": {"type": "char", "readonly": False, "required": False},
    })

    result = validate_payload_against_schema(
        client, "res.partner", "write", args=[[1], {"name": "X", "email": "a@b"}],
    )

    assert result.ok is True
    assert result.errors == []


def test_unknown_field_fails():
    client = _client_with_fields({"name": {"type": "char"}})

    result = validate_payload_against_schema(
        client, "res.partner", "write", args=[[1], {"name": "X", "made_up_field": 42}],
    )

    assert result.ok is False
    assert any("made_up_field" in e for e in result.errors)


def test_readonly_field_fails():
    client = _client_with_fields({
        "name": {"type": "char", "readonly": False},
        "id": {"type": "integer", "readonly": True},
    })

    result = validate_payload_against_schema(
        client, "res.partner", "write", args=[[1], {"id": 42, "name": "X"}],
    )

    assert result.ok is False
    assert any("id" in e and "readonly" in e.lower() for e in result.errors)


def test_empty_fields_response_fails():
    """A {} response indicates a connection failure or missing model — must
    NOT pass validation."""
    client = _client_with_fields({})

    result = validate_payload_against_schema(
        client, "res.partner", "create", args=[{"name": "X"}],
    )

    assert result.ok is False
    assert any("schema" in e.lower() or "fields" in e.lower() for e in result.errors)


def test_create_validates_first_arg_dict():
    client = _client_with_fields({"name": {"type": "char"}})

    result = validate_payload_against_schema(
        client, "res.partner", "create", args=[{"name": "X"}],
    )
    assert result.ok is True


def test_non_dict_payload_skipped():
    """For methods like action_confirm with no vals dict, validation is a no-op."""
    client = _client_with_fields({"name": {"type": "char"}})

    result = validate_payload_against_schema(
        client, "sale.order", "action_confirm", args=[[1]],
    )
    assert result.ok is True


def test_validator_uses_fields_cache():
    """Two calls in quick succession should share one fields_get round-trip."""
    client = _client_with_fields({"name": {"type": "char"}})

    validate_payload_against_schema(
        client, "res.partner", "write", args=[[1], {"name": "A"}],
    )
    validate_payload_against_schema(
        client, "res.partner", "write", args=[[1], {"name": "B"}],
    )
    assert client.execute_kw.call_count == 1
