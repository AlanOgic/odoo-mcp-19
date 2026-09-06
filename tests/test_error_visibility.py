"""Failures must be visible and shared state must be copied safely.

Two classes of defect pinned here:

* ``odoo://model-limitations`` iterated the runtime-issue registry outside its
  lock over a *shallow* snapshot — the nested per-method dicts were still the
  live objects that ``_track_model_issue`` mutates from other threads, so a
  concurrent ``search_read`` fallback could raise ``RuntimeError: dictionary
  changed size during iteration`` in the reader. The snapshot must not alias
  any live container.
* Optional Odoo lookups (``ir.model`` search in ``find-model``, server actions
  in ``actions`` / ``workflows``) and malformed ``args_json`` in a batch used
  to be swallowed by bare ``except ...: pass``. They must be logged.

No live Odoo: ``get_odoo_client`` is patched with a stub.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

import odoo_mcp.resources as resources
from odoo_mcp.constants import RUNTIME_MODEL_ISSUES
from odoo_mcp.safety import classify_batch
from odoo_mcp.utils import _track_model_issue


@pytest.fixture()
def _clean_runtime_issues():
    RUNTIME_MODEL_ISSUES.clear()
    yield
    RUNTIME_MODEL_ISSUES.clear()


def _raising_client(**by_model):
    """Stub whose search_read dispatches on the model: a value is returned, an Exception is raised."""
    client = MagicMock()

    def _search_read(model, *args, **kwargs):
        outcome = by_model.get(model, [])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client.search_read.side_effect = _search_read
    return client


# ----- snapshot isolation -----


def test_runtime_issue_snapshot_does_not_alias_live_nested_dicts(_clean_runtime_issues):
    _track_model_issue("stock.move.line", "search_read", "500 Internal Server Error", domain=[["a.b", "=", 1]])
    live_method = RUNTIME_MODEL_ISSUES["stock.move.line"]["search_read"]

    snapshot = resources._snapshot_runtime_issues()

    snap_method = snapshot["stock.move.line"]["search_read"]
    assert snap_method is not live_method
    assert snap_method["categories"] is not live_method["categories"]
    for category, info in snap_method["categories"].items():
        assert info is not live_method["categories"][category], category
        assert info["domain_patterns"] is not live_method["categories"][category]["domain_patterns"]


def test_model_limitations_reports_from_the_snapshot(_clean_runtime_issues):
    _track_model_issue("stock.move.line", "search_read", "500 Internal Server Error", domain=[["a.b", "=", 1]])
    import json

    payload = json.loads(resources.get_model_limitations())
    assert payload["runtime_detected"]["stock.move.line"]["total_occurrences"] == 1


# ----- swallowed exceptions -----


def test_find_model_logs_when_ir_model_lookup_fails(caplog):
    client = _raising_client(**{"ir.model": ConnectionError("odoo down")})
    caplog.set_level(logging.DEBUG, logger="odoo_mcp.resources")
    with patch.object(resources, "get_odoo_client", return_value=client):
        resources.find_model_resource("zz-no-such-concept")
    assert any("ir.model" in r.getMessage() and "odoo down" in r.getMessage() for r in caplog.records)


def test_discover_actions_logs_when_server_action_lookup_fails(caplog):
    client = _raising_client(**{"ir.actions.server": PermissionError("no access")})
    caplog.set_level(logging.DEBUG, logger="odoo_mcp.resources")
    with patch.object(resources, "get_odoo_client", return_value=client):
        resources.discover_actions_resource("sale.order")
    assert any("ir.actions.server" in r.getMessage() and "no access" in r.getMessage() for r in caplog.records)


def test_workflows_logs_when_server_action_lookup_fails(caplog):
    client = _raising_client(
        **{
            "ir.module.module": [{"name": "sale", "shortdesc": "Sales", "application": True}],
            "ir.actions.server": PermissionError("no access"),
        }
    )
    caplog.set_level(logging.DEBUG, logger="odoo_mcp.resources")
    with patch.object(resources, "get_odoo_client", return_value=client):
        resources.get_workflows()
    assert any("ir.actions.server" in r.getMessage() and "no access" in r.getMessage() for r in caplog.records)


def test_classify_batch_logs_malformed_args_json(caplog):
    caplog.set_level(logging.DEBUG, logger="odoo_mcp.safety")
    classify_batch([{"model": "res.partner", "method": "write", "args_json": "{not json"}])
    assert any("args_json" in r.getMessage() for r in caplog.records)
