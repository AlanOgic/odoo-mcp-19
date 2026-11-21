"""The /doc-bearer cache must never hand one caller's document to another.

Odoo builds ``/doc-bearer/<model>.json`` per caller: it requires the
``api_doc.group_allow_doc`` group, runs ``check_access('read')`` on the model and
lists fields from a user-filtered ``fields_get`` (``api_doc.py``). In multi-user
mode a cache keyed by model name alone served user A's document to user B.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from odoo_mcp import utils
from odoo_mcp.odoo_client import OdooClient

DOC_URL = "https://erp.example.com"


class DocClient:
    """Stands in for one caller's OdooClient; only ``get_model_doc`` is exercised."""

    def __init__(self, username: str, doc: dict | None):
        self.url = DOC_URL
        self.username = username
        self._doc = doc
        self.fetches: list[str] = []

    def get_model_doc(self, model_name: str) -> dict | None:
        self.fetches.append(model_name)
        return self._doc


def _live_doc_as(client: DocClient, model: str) -> dict | None:
    with patch.object(utils, "get_odoo_client", return_value=client):
        return utils._get_live_doc(model)


@pytest.fixture(autouse=True)
def _empty_doc_cache():
    with utils._DOC_CACHE_LOCK:
        utils._DOC_CACHE.clear()
        utils._DOC_REFUSALS.clear()
    yield
    with utils._DOC_CACHE_LOCK:
        utils._DOC_CACHE.clear()
        utils._DOC_REFUSALS.clear()


def test_a_second_identity_fetches_its_own_document():
    admin = DocClient("admin", {"methods": {"compute_sheet": {}}, "fields": {"wage": {}}})
    support = DocClient("support", {"methods": {}, "fields": {}})

    _live_doc_as(admin, "hr.payslip")
    doc = _live_doc_as(support, "hr.payslip")

    assert support.fetches == ["hr.payslip"]
    assert doc == {"methods": {}, "fields": {}}


def test_a_caller_denied_by_odoo_never_receives_a_cached_document():
    """Odoo refuses the doc (no api_doc group, no read access): the caller gets None."""
    admin = DocClient("admin", {"methods": {"compute_sheet": {}}})
    denied = DocClient("support", None)

    _live_doc_as(admin, "hr.payslip")

    assert _live_doc_as(denied, "hr.payslip") is None


def test_the_same_identity_is_served_from_the_cache():
    admin = DocClient("admin", {"methods": {"compute_sheet": {}}})

    _live_doc_as(admin, "hr.payslip")
    _live_doc_as(admin, "hr.payslip")

    assert admin.fetches == ["hr.payslip"]


def test_a_caller_whose_identity_cannot_be_resolved_is_refused():
    """A lost identity or a registry user without stored Odoo credentials must surface,
    not fall back to static data as if the endpoint were merely unavailable."""
    with patch.object(utils, "get_odoo_client", side_effect=PermissionError("no credentials")):
        with pytest.raises(PermissionError, match="no credentials"):
            utils._get_live_doc("hr.payslip")


def test_an_unconfigured_env_client_yields_no_document():
    """Missing ODOO_* config is an unavailable endpoint, not an identity failure."""
    with patch.object(utils, "get_odoo_client", side_effect=FileNotFoundError("no .env")):
        assert utils._get_live_doc("hr.payslip") is None


class Failing(DocClient):
    def get_model_doc(self, model_name: str) -> dict | None:
        self.fetches.append(model_name)
        raise ConnectionError("doc-bearer down")


def test_an_unavailable_endpoint_yields_no_document():
    assert _live_doc_as(Failing("admin", None), "hr.payslip") is None


# ----- negative caching: Odoo's 403 (no api_doc.group_allow_doc) / 404, never a transient failure -----


class _Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def time(self):
        return self.now


@pytest.fixture
def clock():
    fake = _Clock()
    with patch.object(utils, "time", fake):
        yield fake


def test_a_refused_document_is_not_refetched_within_the_negative_ttl(clock):
    denied = DocClient("support", None)
    _live_doc_as(denied, "hr.payslip")
    clock.now += 30

    assert _live_doc_as(denied, "hr.payslip") is None
    assert denied.fetches == ["hr.payslip"]


def test_a_transient_failure_is_retried_on_the_next_call(clock):
    """A timeout or a 5xx must not switch the live @api.private check off for a minute."""
    failing = Failing("admin", None)
    _live_doc_as(failing, "hr.payslip")

    assert _live_doc_as(failing, "hr.payslip") is None
    assert failing.fetches == ["hr.payslip", "hr.payslip"]


def test_odoo_refusing_the_document_is_asked_only_once(clock):
    """End to end through the real OdooClient: api_doc's 403 is recorded, not re-requested."""
    client = OdooClient(url=DOC_URL, db="erp", username="support", api_key="KEY")
    refused = requests.Response()
    refused.status_code, refused._content = 403, b'{"message": "Access Denied"}'
    client.session.post = MagicMock(return_value=refused)
    with patch.object(utils, "get_odoo_client", return_value=client):
        assert utils._get_live_doc("hr.payslip") is None
        assert utils._get_live_doc("hr.payslip") is None
    assert client.session.post.call_count == 1


def test_a_success_without_a_document_is_cached_like_a_refusal(clock):
    """A 200 whose body is not a model doc will not change on retry: it is kept for the negative TTL."""
    odd = DocClient("admin", {"fields": {}})
    _live_doc_as(odd, "hr.payslip")

    assert _live_doc_as(odd, "hr.payslip") is None
    assert odd.fetches == ["hr.payslip"]


def test_refusals_never_evict_cached_documents(clock):
    from odoo_mcp.constants import _DOC_CACHE_MAX_ENTRIES

    admin = DocClient("admin", {"methods": {"compute_sheet": {}}})
    denied = DocClient("support", None)
    _live_doc_as(admin, "hr.payslip")
    for i in range(_DOC_CACHE_MAX_ENTRIES + 20):
        clock.now += 1
        _live_doc_as(denied, f"x.unknown{i}")

    _live_doc_as(admin, "hr.payslip")

    assert admin.fetches == ["hr.payslip"]
    assert len(utils._DOC_REFUSALS) == _DOC_CACHE_MAX_ENTRIES


def test_an_unavailable_document_is_retried_once_the_negative_ttl_expires(clock):
    from odoo_mcp.constants import _DOC_NEGATIVE_TTL

    client = DocClient("support", None)
    _live_doc_as(client, "hr.payslip")
    client._doc = {"methods": {"compute_sheet": {}}}
    clock.now += _DOC_NEGATIVE_TTL + 1

    assert _live_doc_as(client, "hr.payslip") == {"methods": {"compute_sheet": {}}}
    assert client.fetches == ["hr.payslip", "hr.payslip"]


def test_one_identity_denied_does_not_deny_another(clock):
    denied = DocClient("support", None)
    admin = DocClient("admin", {"methods": {"compute_sheet": {}}})

    _live_doc_as(denied, "hr.payslip")

    assert _live_doc_as(admin, "hr.payslip") == {"methods": {"compute_sheet": {}}}


def test_the_server_version_of_an_unresolvable_caller_is_refused():
    """Same identity rule as the documents: a lost identity surfaces, it is not read as 'version unknown'."""
    with patch.object(utils, "get_odoo_client", side_effect=PermissionError("no credentials")):
        with pytest.raises(PermissionError, match="no credentials"):
            utils.get_server_version()
