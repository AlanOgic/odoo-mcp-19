"""Tests for the live fields_get cache used by payload pre-flight."""
import time
from unittest.mock import MagicMock

import pytest

from odoo_mcp.utils import (
    _FIELDS_CACHE,
    _FIELDS_CACHE_LOCK,
    get_fields_for_model,
)


@pytest.fixture(autouse=True)
def clear_cache():
    with _FIELDS_CACHE_LOCK:
        _FIELDS_CACHE.clear()
    yield
    with _FIELDS_CACHE_LOCK:
        _FIELDS_CACHE.clear()


def test_first_call_hits_odoo():
    client = MagicMock()
    client.execute_method.return_value = {
        "name": {"type": "char", "readonly": False, "required": True},
        "id": {"type": "integer", "readonly": True, "required": False},
    }

    fields = get_fields_for_model(client, "res.partner")

    assert "name" in fields
    assert client.execute_method.call_count == 1


def test_second_call_hits_cache():
    client = MagicMock()
    client.execute_method.return_value = {"name": {"type": "char"}}

    get_fields_for_model(client, "res.partner")
    get_fields_for_model(client, "res.partner")

    assert client.execute_method.call_count == 1


def test_cache_expires_after_ttl(monkeypatch):
    client = MagicMock()
    client.execute_method.return_value = {"name": {"type": "char"}}

    get_fields_for_model(client, "res.partner")

    # Fast-forward time past the TTL (60s).
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 61)

    get_fields_for_model(client, "res.partner")

    assert client.execute_method.call_count == 2


def test_empty_response_not_cached():
    """An Odoo connection that returns {} should not poison the cache."""
    client = MagicMock()
    client.execute_method.return_value = {}

    fields = get_fields_for_model(client, "res.partner")
    assert fields == {}

    # Second call must re-query (we don't trust an empty response).
    client.execute_method.return_value = {"name": {"type": "char"}}
    fields = get_fields_for_model(client, "res.partner")
    assert "name" in fields
    assert client.execute_method.call_count == 2


def test_lru_eviction():
    """Cache caps at 100 entries; oldest evicted first."""
    client = MagicMock()
    client.execute_method.return_value = {"x": {"type": "char"}}

    for i in range(110):
        get_fields_for_model(client, f"model.test_{i}")

    with _FIELDS_CACHE_LOCK:
        assert len(_FIELDS_CACHE) <= 100
