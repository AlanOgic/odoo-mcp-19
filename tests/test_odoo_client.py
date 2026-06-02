"""Unit tests for the JSON-2 OdooClient transport layer.

These pin two corrections derived from Odoo 19's External JSON-2 API
(``content/developer/reference/external_api.rst``):

  1. Authentication is a bearer API key only. The client must always send an
     ``Authorization: Bearer <credential>`` header; a missing header (the old
     password-only path) silently produced unauthenticated requests.
  2. The success response body *is* the bare serialized return value — there is
     no ``{"result": ...}`` envelope (that was the legacy ``/jsonrpc``
     convention). The client must not unwrap a ``result`` key, or a method that
     legitimately returns a dict containing ``result`` gets corrupted.
"""

from unittest.mock import MagicMock

import pytest

from odoo_mcp.odoo_client import OdooClient


def _make_client(**overrides):
    params = dict(
        url="https://mycompany.example.com",
        db="mycompany",
        username="bot",
        api_key="THEKEY",
    )
    params.update(overrides)
    return OdooClient(**params)


def _stub_response(json_value, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_value
    return resp


class TestAuthHeader:
    """JSON-2 only supports bearer API-key auth; a header must always be sent."""

    def test_api_key_sets_bearer_header(self):
        client = _make_client(api_key="THEKEY")
        assert client.session.headers["Authorization"] == "Bearer THEKEY"

    def test_password_only_still_sets_bearer_header(self):
        # Previously this path set no Authorization header at all → silent
        # unauthenticated requests. The value must at least be sent so Odoo can
        # return a clear 401 if it isn't a valid key.
        client = _make_client(api_key=None, password="PWVALUE")
        assert client.session.headers["Authorization"] == "Bearer PWVALUE"

    def test_password_fallback_warns(self, caplog):
        with caplog.at_level("WARNING"):
            _make_client(api_key=None, password="PWVALUE")
        assert any("JSON-2" in r.message or "API key" in r.message for r in caplog.records)

    def test_no_credential_raises(self):
        with pytest.raises(ValueError):
            _make_client(api_key=None, password=None)


class TestResponseNotUnwrapped:
    """The bare return value must be passed through, never unwrapped."""

    def test_dict_with_result_key_is_returned_intact(self):
        client = _make_client()
        payload = {"result": {"foo": 1}, "other": 2}
        client.session = MagicMock()
        client.session.post.return_value = _stub_response(payload)

        out = client._execute("res.partner", "some_model_method")

        assert out == payload  # NOT unwrapped to {"foo": 1}

    def test_list_return_passes_through(self):
        client = _make_client()
        client.session = MagicMock()
        client.session.post.return_value = _stub_response([1, 2, 3])

        assert client._execute("res.partner", "search") == [1, 2, 3]

    def test_plain_dict_passes_through(self):
        client = _make_client()
        client.session = MagicMock()
        client.session.post.return_value = _stub_response({"name": "Acme", "id": 5})

        assert client._execute("res.partner", "read") == {"name": "Acme", "id": 5}


class TestErrorBodyNeverLeaksTraceback:
    """A 4xx/5xx body carries a 'debug' traceback that must not reach the client."""

    def test_debug_traceback_not_in_raised_message(self):
        client = _make_client()
        client.session = MagicMock()
        client.session.post.return_value = _stub_response(
            {
                "name": "werkzeug.exceptions.Unauthorized",
                "message": "Invalid apikey",
                "debug": "Traceback (most recent call last): SECRET_INTERNALS",
            },
            status_code=401,
        )

        with pytest.raises(ValueError) as exc:
            client._execute("res.partner", "read", [1])

        assert "Invalid apikey" in str(exc.value)
        assert "SECRET_INTERNALS" not in str(exc.value)
