"""The /doc-bearer cache must never hand one caller's document to another.

Odoo builds ``/doc-bearer/<model>.json`` per caller: it requires the
``api_doc.group_allow_doc`` group, runs ``check_access('read')`` on the model and
lists fields from a user-filtered ``fields_get`` (``api_doc.py``). In multi-user
mode a cache keyed by model name alone served user A's document to user B.
"""

from unittest.mock import patch

import pytest

from odoo_mcp import utils

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
    yield
    with utils._DOC_CACHE_LOCK:
        utils._DOC_CACHE.clear()


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


def test_an_unavailable_endpoint_yields_no_document():
    class Failing(DocClient):
        def get_model_doc(self, model_name: str) -> dict | None:
            raise ConnectionError("doc-bearer down")

    assert _live_doc_as(Failing("admin", None), "hr.payslip") is None
