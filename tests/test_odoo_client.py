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

from unittest.mock import MagicMock, patch

import pytest
import requests

from odoo_mcp.constants import RATE_LIMIT_RETRY_DEFAULT_DELAY, RATE_LIMIT_RETRY_MAX_DELAY
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


class TestFieldsGetAttributes:
    """``get_model_fields`` narrows the request when an attribute subset is given.

    A full ``fields_get`` on a big model is ~300 KB; the compact schema views only
    read a handful of attributes, so forwarding ``attributes`` to JSON-2 is what
    makes ``odoo://bundle`` and ``session-bootstrap`` cheap.
    """

    def test_attributes_are_forwarded_to_fields_get(self):
        client = _make_client()
        client.session = MagicMock()
        client.session.post.return_value = _stub_response({"name": {"type": "char"}})

        client.get_model_fields("res.partner", attributes=["type", "required"])

        payload = client.session.post.call_args.kwargs["json"]
        assert payload == {"attributes": ["type", "required"]}

    def test_no_attributes_requests_the_full_definition(self):
        client = _make_client()
        client.session = MagicMock()
        client.session.post.return_value = _stub_response({"name": {"type": "char"}})

        client.get_model_fields("res.partner")

        assert client.session.post.call_args.kwargs["json"] == {}


class TestRateLimitRetry:
    """Odoo SaaS answers a burst of requests with 429 Too Many Requests.

    A single retry after the ``Retry-After`` delay turns a transient throttle
    into a successful call instead of an ``errors`` entry in session-bootstrap.
    """

    @staticmethod
    def _rate_limited(retry_after=None):
        resp = _stub_response({"message": "Too Many Requests"}, status_code=429)
        resp.reason = "Too Many Requests"
        resp.url = "https://mycompany.example.com/json/2/res.partner/fields_get"
        resp.headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
        return resp

    def test_retries_once_after_429_and_returns_the_result(self):
        client = _make_client()
        client.session.post = MagicMock(side_effect=[self._rate_limited(retry_after=2), _stub_response({"id": 1})])
        with patch("odoo_mcp.odoo_client.time.sleep") as sleep:
            result = client.execute_method("res.partner", "read", [1])
        assert result == {"id": 1}
        assert client.session.post.call_count == 2
        sleep.assert_called_once_with(2.0)

    def test_429_without_retry_after_header_waits_the_default_delay(self):
        client = _make_client()
        client.session.post = MagicMock(side_effect=[self._rate_limited(), _stub_response([])])
        with patch("odoo_mcp.odoo_client.time.sleep") as sleep:
            client.execute_method("res.partner", "read", [1])
        sleep.assert_called_once_with(RATE_LIMIT_RETRY_DEFAULT_DELAY)

    def test_retry_after_is_capped_so_a_hostile_header_cannot_stall_the_server(self):
        client = _make_client()
        client.session.post = MagicMock(side_effect=[self._rate_limited(retry_after=3600), _stub_response([])])
        with patch("odoo_mcp.odoo_client.time.sleep") as sleep:
            client.execute_method("res.partner", "read", [1])
        sleep.assert_called_once_with(RATE_LIMIT_RETRY_MAX_DELAY)

    def test_unparseable_retry_after_falls_back_to_the_default_delay(self):
        client = _make_client()
        client.session.post = MagicMock(
            side_effect=[self._rate_limited(retry_after="Wed, 21 Oct 2026 07:28:00 GMT"), _stub_response([])]
        )
        with patch("odoo_mcp.odoo_client.time.sleep") as sleep:
            client.execute_method("res.partner", "read", [1])
        sleep.assert_called_once_with(RATE_LIMIT_RETRY_DEFAULT_DELAY)

    def test_second_429_is_raised_not_retried_forever(self):
        client = _make_client()
        client.session.post = MagicMock(
            side_effect=[self._rate_limited(retry_after=1), self._rate_limited(retry_after=1)]
        )
        with patch("odoo_mcp.odoo_client.time.sleep"), pytest.raises(ValueError, match="429"):
            client.execute_method("res.partner", "read", [1])
        assert client.session.post.call_count == 2

    def test_other_http_errors_are_not_retried(self):
        client = _make_client()
        failed = _stub_response({"message": "boom"}, status_code=500)
        failed.reason = "Internal Server Error"
        failed.url = "https://mycompany.example.com/json/2/res.partner/read"
        client.session.post = MagicMock(return_value=failed)
        with patch("odoo_mcp.odoo_client.time.sleep") as sleep, pytest.raises(ValueError, match="500"):
            client.execute_method("res.partner", "read", [1])
        assert client.session.post.call_count == 1
        sleep.assert_not_called()

    def test_side_effect_methods_are_never_retried_after_429(self):
        """A 429 may be raised after partial processing on some stacks — never re-send a write."""
        client = _make_client()
        client.session.post = MagicMock(side_effect=[self._rate_limited(retry_after=1), _stub_response([])])
        with patch("odoo_mcp.odoo_client.time.sleep") as sleep, pytest.raises(ValueError, match="429"):
            client.execute_method("res.partner", "unlink", [1])
        assert client.session.post.call_count == 1
        sleep.assert_not_called()

    def test_connection_error_on_the_retry_surfaces_as_connection_error(self):
        import requests

        client = _make_client()
        client.session.post = MagicMock(
            side_effect=[self._rate_limited(retry_after=1), requests.exceptions.ConnectionError("down")]
        )
        with patch("odoo_mcp.odoo_client.time.sleep"), pytest.raises(ConnectionError):
            client.execute_method("res.partner", "read", [1])


def _http_response(status_code, body=b"{}"):
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = body
    resp.url = "https://mycompany.example.com/doc-bearer/sale.order.json"
    return resp


_DOC_FETCHES = pytest.mark.parametrize(
    "fetch", [lambda c: c.get_model_doc("sale.order"), lambda c: c.get_api_index()], ids=["model-doc", "index"]
)


class TestDocEndpointsSeparateRefusalFromFailure:
    """/doc-bearer/ refuses with 403 (no api_doc group, no read access) or 404 (unknown model):
    a definitive answer the caller may cache (api_doc/controllers/api_doc.py @ 19.0). Anything
    else can succeed on the next try, so it raises instead of looking like a refusal."""

    @_DOC_FETCHES
    @pytest.mark.parametrize("status", [403, 404])
    def test_a_refusal_returns_none(self, fetch, status):
        client = _make_client()
        client.session.post = MagicMock(return_value=_http_response(status))
        assert fetch(client) is None

    @_DOC_FETCHES
    @pytest.mark.parametrize("status", [429, 500, 503])
    def test_a_transient_http_error_raises(self, fetch, status):
        client = _make_client()
        client.session.post = MagicMock(return_value=_http_response(status))
        with pytest.raises(requests.HTTPError):
            fetch(client)

    @_DOC_FETCHES
    def test_a_network_error_raises(self, fetch):
        client = _make_client()
        client.session.post = MagicMock(side_effect=requests.exceptions.ConnectTimeout("slow"))
        with pytest.raises(requests.exceptions.ConnectTimeout):
            fetch(client)


def _version_response(*version_info):
    return _stub_response({"version_info": list(version_info), "version": "x"})


class TestServerVersion:
    """The connected series, from ``GET /web/version`` (rpc addon, auth='none'): ``release.version_info``."""

    @pytest.mark.parametrize(
        ("version_info", "expected"),
        [
            ((19, 0, 0, "final", 0, ""), (19, 0)),
            (("saas~19", 3, 0, "final", 0, ""), (19, 3)),
            ((20, 0, 0, "final", 0, ""), (20, 0)),
            ((20, 1, 0, "alpha", 1, ""), (20, 1)),
        ],
    )
    def test_version_info_gives_the_series(self, version_info, expected):
        client = _make_client()
        client.session.get = MagicMock(return_value=_version_response(*version_info))
        assert client.server_version() == expected
        assert client.session.get.call_args.args[0] == "https://mycompany.example.com/web/version"

    def test_the_version_is_read_once(self):
        client = _make_client()
        client.session.get = MagicMock(return_value=_version_response("saas~19", 3, 0, "final", 0, ""))
        client.server_version()
        client.server_version()
        assert client.session.get.call_count == 1

    def test_an_unreadable_version_is_none_and_retried(self):
        client = _make_client()
        client.session.get = MagicMock(side_effect=requests.exceptions.ConnectionError("down"))
        assert client.server_version() is None
        client.session.get = MagicMock(return_value=_version_response(19, 0, 0, "final", 0, ""))
        assert client.server_version() == (19, 0)

    def test_the_version_is_read_again_after_the_ttl(self):
        """An Odoo Online upgrade (19.3 → 19.4) must reach a long-running server without a restart."""
        from odoo_mcp.constants import SERVER_VERSION_TTL

        client = _make_client()
        client.session.get = MagicMock(return_value=_version_response("saas~19", 3, 0, "final", 0, ""))
        with patch("odoo_mcp.odoo_client.time.time", return_value=1_000.0):
            client.server_version()
        client.session.get = MagicMock(return_value=_version_response("saas~19", 4, 0, "final", 0, ""))
        with patch("odoo_mcp.odoo_client.time.time", return_value=1_000.0 + SERVER_VERSION_TTL + 1):
            assert client.server_version() == (19, 4)

    def test_a_removed_method_is_refused_before_it_is_sent(self):
        client = _make_client()
        client.session.get = MagicMock(return_value=_version_response("saas~19", 3, 0, "final", 0, ""))
        client.session.post = MagicMock()
        with pytest.raises(ValueError, match="formatted_read_group"):
            client.execute_method("sale.order", "read_group", [], ["amount_total"], ["partner_id"])
        client.session.post.assert_not_called()

    def test_read_group_on_20_is_sent_with_the_20_signature(self):
        client = _make_client()
        client.session.get = MagicMock(return_value=_version_response(20, 0, 0, "final", 0, ""))
        client.session.post = MagicMock(return_value=_stub_response([]))
        client.execute_method("sale.order", "read_group", [], ["partner_id"], ["amount_total:sum"])
        assert client.session.post.call_args.kwargs["json"] == {
            "domain": [],
            "groupby": ["partner_id"],
            "aggregates": ["amount_total:sum"],
        }

    def test_an_unchanged_method_does_not_read_the_version(self):
        client = _make_client()
        client.session.get = MagicMock()
        client.session.post = MagicMock(return_value=_stub_response([]))
        client.execute_method("res.partner", "search", [])
        client.session.get.assert_not_called()
