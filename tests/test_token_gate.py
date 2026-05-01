"""
Unit tests for the confirmation token gate (server._issue/_validate_confirmation_token).

Covers C1: tokens are bound to (model, method, payload_digest) so that the
confirmed=true re-call must reproduce the exact same args/kwargs that the gate
saw at issue time. Substituting a different payload after the user confirmed
the original payload is rejected.
"""

from __future__ import annotations

import time

import pytest

from odoo_mcp import server


@pytest.fixture(autouse=True)
def _clear_token_store():
    """Each test starts with an empty token store."""
    with server._CONFIRMATION_LOCK:
        server._CONFIRMATION_TOKENS.clear()
    yield
    with server._CONFIRMATION_LOCK:
        server._CONFIRMATION_TOKENS.clear()


# ----- _payload_digest -----


class TestPayloadDigest:
    def test_deterministic_for_same_input(self):
        a = server._payload_digest({"args": [[1, 2]], "kwargs": {"limit": 10}})
        b = server._payload_digest({"args": [[1, 2]], "kwargs": {"limit": 10}})
        assert a == b

    def test_dict_key_order_does_not_matter(self):
        a = server._payload_digest({"a": 1, "b": 2})
        b = server._payload_digest({"b": 2, "a": 1})
        assert a == b, "sort_keys=True should make dict ordering irrelevant"

    def test_list_order_matters(self):
        a = server._payload_digest([1, 2, 3])
        b = server._payload_digest([3, 2, 1])
        assert a != b, "list order is significant — operation order in a batch matters"

    def test_extra_id_changes_digest(self):
        """The C1 attack: extending args[0] with extra IDs must change the digest."""
        original = server._payload_digest({"args": [[1]], "kwargs": {}})
        extended = server._payload_digest({"args": [[1, 2, 3]], "kwargs": {}})
        assert original != extended

    def test_changed_kwarg_changes_digest(self):
        a = server._payload_digest({"args": [], "kwargs": {"name": "Alice"}})
        b = server._payload_digest({"args": [], "kwargs": {"name": "Bob"}})
        assert a != b


# ----- token issue + validate happy path -----


class TestTokenHappyPath:
    def test_matching_call_validates(self):
        digest = server._payload_digest({"args": [[1]], "kwargs": {}})
        token = server._issue_confirmation_token("res.partner", "unlink", digest)

        err = server._validate_confirmation_token(token, "res.partner", "unlink", digest)
        assert err is None

    def test_token_is_single_use(self):
        digest = server._payload_digest({"args": [[1]], "kwargs": {}})
        token = server._issue_confirmation_token("res.partner", "unlink", digest)

        assert server._validate_confirmation_token(token, "res.partner", "unlink", digest) is None
        # Second use is rejected — token already consumed
        err = server._validate_confirmation_token(token, "res.partner", "unlink", digest)
        assert err is not None
        assert "invalid or already used" in err


# ----- C1: payload binding closes the bypass -----


class TestPayloadBinding:
    def test_extra_id_rejected(self):
        """C1 attack vector: token issued for unlink([1]) must NOT validate for unlink([1, 2, 3])."""
        original_args = [[1]]
        attack_args = [[1, 2, 3, 4, 5]]
        issue_digest = server._payload_digest({"args": original_args, "kwargs": {}})
        consume_digest = server._payload_digest({"args": attack_args, "kwargs": {}})

        token = server._issue_confirmation_token("res.partner", "unlink", issue_digest)
        err = server._validate_confirmation_token(token, "res.partner", "unlink", consume_digest)

        assert err is not None
        assert "different payload" in err

    def test_changed_vals_rejected(self):
        """write([1], {name: 'Alice'}) gate cannot be reused for write([1], {name: 'Evil'})."""
        issue = server._payload_digest({"args": [[1], {"name": "Alice"}], "kwargs": {}})
        consume = server._payload_digest({"args": [[1], {"name": "Evil"}], "kwargs": {}})

        token = server._issue_confirmation_token("res.partner", "write", issue)
        err = server._validate_confirmation_token(token, "res.partner", "write", consume)

        assert err is not None
        assert "different payload" in err

    def test_added_kwarg_rejected(self):
        """Adding a kwarg (e.g. force=True) on the re-call invalidates the token."""
        issue = server._payload_digest({"args": [[1]], "kwargs": {}})
        consume = server._payload_digest({"args": [[1]], "kwargs": {"force": True}})

        token = server._issue_confirmation_token("sale.order", "action_confirm", issue)
        err = server._validate_confirmation_token(token, "sale.order", "action_confirm", consume)

        assert err is not None
        assert "different payload" in err


# ----- mismatched (model, method) still rejected -----


class TestModelMethodBinding:
    def test_different_method_rejected(self):
        digest = server._payload_digest({"args": [[1]], "kwargs": {}})
        token = server._issue_confirmation_token("res.partner", "write", digest)

        err = server._validate_confirmation_token(token, "res.partner", "unlink", digest)
        assert err is not None
        assert "issued for res.partner.write" in err

    def test_different_model_rejected(self):
        digest = server._payload_digest({"args": [[1]], "kwargs": {}})
        token = server._issue_confirmation_token("res.partner", "unlink", digest)

        err = server._validate_confirmation_token(token, "account.move", "unlink", digest)
        assert err is not None
        assert "issued for res.partner.unlink" in err


# ----- TTL handling -----


class TestTokenExpiry:
    def test_expired_token_rejected(self, monkeypatch):
        digest = server._payload_digest({"args": [[1]], "kwargs": {}})
        token = server._issue_confirmation_token("res.partner", "unlink", digest)

        # Fast-forward time past the TTL by patching time.time used in server module
        original = time.time()
        monkeypatch.setattr(
            server.time, "time", lambda: original + server._CONFIRMATION_TTL + 1
        )

        err = server._validate_confirmation_token(token, "res.partner", "unlink", digest)
        assert err is not None
        assert "expired" in err.lower()

    def test_missing_token_rejected(self):
        digest = server._payload_digest({"args": [], "kwargs": {}})
        err = server._validate_confirmation_token(None, "res.partner", "unlink", digest)
        assert err is not None
        assert "requires a confirmation_token" in err


# ----- Batch token (C1 batch variant) -----


class TestBatchTokenBinding:
    def test_same_count_different_ops_rejected(self):
        """C1 batch attack: 5 benign ops gate a token that an attacker
        tries to reuse for 5 evil ops. Must be rejected by digest mismatch."""
        benign_ops = [
            {"model": "res.partner", "method": "write", "args_json": "[[1], {}]"},
        ] * 5
        evil_ops = [
            {"model": "res.partner", "method": "unlink", "args_json": "[[1,2,3,4,5]]"},
        ] * 5

        issue = server._payload_digest(benign_ops)
        consume = server._payload_digest(evil_ops)
        assert issue != consume

        token = server._issue_confirmation_token("__batch__", "batch", issue)
        err = server._validate_confirmation_token(token, "__batch__", "batch", consume)
        assert err is not None
        assert "different payload" in err

    def test_identical_batch_validates(self):
        ops = [
            {"model": "res.partner", "method": "write", "args_json": "[[1], {}]"},
            {"model": "sale.order", "method": "action_confirm", "args_json": "[[15]]"},
        ]
        digest = server._payload_digest(ops)
        token = server._issue_confirmation_token("__batch__", "batch", digest)

        err = server._validate_confirmation_token(token, "__batch__", "batch", digest)
        assert err is None


# ----- Workflow token (C1 workflow variant) -----


class TestWorkflowTokenBinding:
    def test_different_params_rejected(self):
        """Token for quote_to_cash with order_id=15 cannot be reused for order_id=99."""
        issue = server._payload_digest({"order_id": 15})
        consume = server._payload_digest({"order_id": 99})

        token = server._issue_confirmation_token("__workflow__", "quote_to_cash", issue)
        err = server._validate_confirmation_token(token, "__workflow__", "quote_to_cash", consume)
        assert err is not None
        assert "different payload" in err
