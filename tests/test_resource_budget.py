"""read_resource keeps an oversized payload valid JSON, and session-bootstrap fits its default budget.

Audit finding R1 (2026-11): ten resources exceeded the 15 000-char bridge budget and were cut at a
character offset, so Claude Desktop / claude.ai received a fragment no parser accepts. The first
resource an agent reads, ``odoo://session-bootstrap``, was 60 KB.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from odoo_mcp import resources
from odoo_mcp.constants import _READ_RESOURCE_MAX_CHARS
from odoo_mcp.server import read_resource
from odoo_mcp.utils import _build_bootstrap_schema, _dumps, fit_json_to_budget

HINT = "narrow it"


def _fit(payload, max_chars):
    out = fit_json_to_budget(_dumps(payload), max_chars, HINT)
    assert out is not None
    return out, json.loads(out)


class TestFitJsonToBudget:
    def test_the_result_parses_and_fits(self):
        out, data = _fit({"model": "x", "fields": {f"field_{i}": {"t": "char"} for i in range(500)}}, 2000)

        assert len(out) <= 2000
        assert data["_truncated"] is True
        assert data["_hint"] == HINT

    def test_the_marker_reports_the_original_size_and_what_was_dropped(self):
        payload = {"model": "x", "fields": {f"field_{i}": {"t": "char"} for i in range(500)}}

        _, data = _fit(payload, 2000)

        assert data["_total_chars"] == len(_dumps(payload))
        (dropped,) = data["_dropped"]
        assert dropped == {"path": "fields", "kept": len(data["fields"]), "of": 500}
        assert 0 < dropped["kept"] < 500

    def test_the_head_of_the_collection_survives_in_order(self):
        _, data = _fit({"names": [f"model.{i:04d}" for i in range(2000)]}, 1500)

        kept = data["names"]
        assert kept == [f"model.{i:04d}" for i in range(len(kept))]

    def test_keys_beside_the_dominant_collection_are_untouched(self):
        _, data = _fit({"model": "sale.order", "field_count": 500, "fields": {f"f{i}": i for i in range(5000)}}, 1000)

        assert data["model"] == "sale.order"
        assert data["field_count"] == 500

    def test_a_nested_dominant_collection_is_named_by_its_path(self):
        payload = {"schemas": {"sale.order": {"fields": {f"f{i}": {"t": "char"} for i in range(900)}, "n": 1}}}

        _, data = _fit(payload, 1500)

        assert [d["path"] for d in data["_dropped"]] == ["schemas/sale.order/fields"]

    def test_several_collections_are_trimmed_when_one_is_not_enough(self):
        # odoo://models: the details map is emptied first, then the name list is shortened.
        names = [f"model.name.{i:04d}" for i in range(1000)]
        payload = {"model_names": names, "models_details": {n: {"name": n.upper()} for n in names}}

        out, data = _fit(payload, 5000)

        assert len(out) <= 5000
        dropped = {d["path"]: d for d in data["_dropped"]}
        assert dropped["models_details"] == {"path": "models_details", "kept": 0, "of": 1000}
        assert dropped["model_names"]["of"] == 1000
        assert data["model_names"] == names[: dropped["model_names"]["kept"]]

    def test_a_small_overflow_costs_a_few_items_not_a_whole_sibling(self):
        payload = {"a": ["x" * 20] * 700, "b": ["y" * 20] * 700, "count": 3, "hint": "h"}
        budget = len(_dumps(payload)) - 100

        out, data = _fit(payload, budget)

        assert len(out) > budget * 0.95
        assert data["count"] == 3 and data["hint"] == "h"
        assert 1380 < len(data["a"]) + len(data["b"]) < 1400

    def test_sections_far_bigger_than_their_siblings_are_trimmed_before_any_key_is_lost(self):
        # odoo://module-knowledge: two heavy sections among many light ones.
        payload = {f"section_{i}": "x" * 300 for i in range(10)}
        payload["orm_methods"] = {f"method_{i}": "d" * 80 for i in range(200)}
        payload["tail"] = "kept"

        out, data = _fit(payload, 6000)

        assert len(out) > 6000 * 0.9
        assert set(payload) <= set(data)
        assert [d["path"] for d in data["_dropped"]] == ["orm_methods"]

    def test_similar_models_lose_the_last_one_then_part_of_the_next(self):
        model = {"fields": {f"field_{i}": {"t": "char"} for i in range(200)}}
        payload = {"schemas": {name: json.loads(_dumps(model)) for name in ("a", "b", "c")}}
        budget = len(_dumps(payload)) // 2

        out, data = _fit(payload, budget)

        assert len(out) > budget * 0.9
        assert list(data["schemas"]) == ["a", "b"]
        # The marker grows by one entry on the last pass, which costs "a" a handful of fields.
        assert len(data["schemas"]["a"]["fields"]) > 190
        assert 0 < len(data["schemas"]["b"]["fields"]) < 200

    def test_a_trimmed_collection_later_removed_whole_is_not_reported_twice(self):
        payload = {"schemas": {"a": {"f": ["x" * 9] * 60}, "b": {"f": ["x" * 9] * 100}}, "z": "q" * 900}

        _, data = _fit(payload, 700)

        paths = [d["path"] for d in data["_dropped"]]
        present = {"(root)"} | {p for p in paths if p.split("/")[0] in data}
        assert set(paths) <= present

    def test_a_payload_key_cannot_overwrite_the_marker(self):
        _, data = _fit({"_truncated": "no", "_hint": "mine", "rows": list(range(5000))}, 800)

        assert data["_truncated"] is True and data["_hint"] == HINT

    def test_no_dominant_child_drops_trailing_top_level_keys(self):
        payload = {f"section_{i}": "x" * 400 for i in range(20)}

        _, data = _fit(payload, 3000)

        (dropped,) = data["_dropped"]
        assert dropped["path"] == "(root)" and dropped["of"] == 20
        assert [k for k in data if not k.startswith("_")] == [f"section_{i}" for i in range(dropped["kept"])]

    def test_a_collection_trimmed_twice_keeps_its_original_count(self):
        # The second pass (the marker grew) must not report the first pass's remainder as the total.
        payload = {"items": ["x" * 90 for _ in range(300)]}

        _, data = _fit(payload, 1000)

        assert [d["of"] for d in data["_dropped"]] == [300]

    def test_a_top_level_list_is_wrapped(self):
        _, data = _fit([{"id": i} for i in range(2000)], 1000)

        assert data["_truncated"] is True
        assert data["items"][0] == {"id": 0}

    def test_a_budget_smaller_than_the_marker_still_returns_valid_json(self):
        out, data = _fit({"a": list(range(1000))}, 10)

        assert data["_truncated"] is True
        assert [key for key in data if not key.startswith("_")] == []

    def test_text_that_is_not_json_is_left_to_the_caller(self):
        assert fit_json_to_budget("plain text " * 100, 50, HINT) is None

    @pytest.mark.parametrize("max_chars", [300, 1000, 5000, 15000])
    def test_one_oversized_string_cannot_break_the_json(self, max_chars):
        out, data = _fit({"note": "n" * 40000, "rows": list(range(3000))}, max_chars)

        assert data["_truncated"] is True
        assert len(out) <= max_chars
        assert "note" not in data  # the string alone is over every budget here
        assert len(data.get("rows", [])) == 3000 or max_chars < 15000


class TestReadResourceBudget:
    def test_an_oversized_resource_comes_back_as_valid_json_within_the_default_budget(self):
        full = read_resource("odoo://module-knowledge", max_chars=0)
        assert len(full) > _READ_RESOURCE_MAX_CHARS  # otherwise this test proves nothing

        out = read_resource("odoo://module-knowledge")

        data = json.loads(out)
        assert len(out) <= _READ_RESOURCE_MAX_CHARS
        assert data["_truncated"] is True
        assert data["_total_chars"] == len(full)
        assert "max_chars=0" in data["_hint"]

    def test_max_chars_zero_returns_the_payload_untouched(self):
        data = json.loads(read_resource("odoo://module-knowledge", max_chars=0))

        assert "_truncated" not in data
        assert "error_patterns" in data  # the last section, the first one a cut removes

    def test_a_payload_within_the_budget_carries_no_marker(self):
        out = read_resource("odoo://domain-syntax")

        assert len(out) <= _READ_RESOURCE_MAX_CHARS
        assert "_truncated" not in json.loads(out)


def _field(i: int) -> tuple[str, dict]:
    """A field shaped like the ones measured on Odoo 19.3: ~22-char names, a third relational."""
    name = f"x_studio_business_field_{i:03d}"
    if i % 3 == 0:
        return name, {"type": "many2one", "string": "F", "relation": "account.payment.method.line"}
    return name, {"type": "char", "string": "F", "readonly": i % 5 == 0}


_BIG_MODEL = {
    "id": {"type": "integer", "string": "ID", "readonly": True},
    "name": {"type": "char", "string": "Name", "required": True},
    "state": {"type": "selection", "string": "Status", "readonly": True, "selection": [["draft", "Draft"]]},
    **dict(_field(i) for i in range(300)),
}


class TestBootstrapSchema:
    FIELDS = {
        "id": {"type": "integer", "readonly": True},
        "street": {"type": "char"},
        "partner_id": {"type": "many2one", "relation": "res.partner", "required": True},
        "order_line": {"type": "one2many", "relation": "sale.order.line"},
        "tag_ids": {"type": "many2many", "relation": "crm.tag"},
        "amount_total": {"type": "monetary", "readonly": True},
        "message_follower_ids": {"type": "one2many", "relation": "mail.followers"},
        "activity_ids": {"type": "one2many", "relation": "mail.activity"},
        "access_token": {"type": "char"},
        "state": {"type": "selection", "readonly": True, "selection": [["draft", "Quotation"], ["sale", "Order"]]},
    }

    def test_writable_fields_are_grouped_by_type_with_their_comodel(self):
        schema = _build_bootstrap_schema(self.FIELDS, 50)

        assert schema["writable"] == {
            "many2one": {"partner_id": "res.partner"},
            "char": ["street"],
            "one2many": {"order_line": "sale.order.line"},
            "many2many": {"tag_ids": "crm.tag"},
        }

    def test_readonly_and_mixin_fields_are_left_out(self):
        listed = json.dumps(_build_bootstrap_schema(self.FIELDS, 50)["writable"])

        for absent in ("amount_total", "message_follower_ids", "activity_ids", "access_token", '"id"'):
            assert absent not in listed

    def test_a_readonly_state_still_gives_its_values(self):
        assert _build_bootstrap_schema(self.FIELDS, 50)["state_values"] == [["draft", "Quotation"], ["sale", "Order"]]

    def test_required_fields_come_first_when_the_cap_bites(self):
        schema = _build_bootstrap_schema(self.FIELDS, 1)

        assert schema["required"] == ["partner_id"]
        assert schema["writable"] == {"many2one": {"partner_id": "res.partner"}}
        assert schema["writable_shown"] == 1

    def test_the_cap_follows_odoo_field_order(self):
        schema = _build_bootstrap_schema(self.FIELDS, 2)

        assert schema["writable"] == {"many2one": {"partner_id": "res.partner"}, "char": ["street"]}

    def test_counts_say_whether_the_writable_list_is_complete(self):
        schema = _build_bootstrap_schema(self.FIELDS, 2)

        assert (schema["field_count"], schema["writable_total"], schema["writable_shown"]) == (len(self.FIELDS), 4, 2)

    def test_a_required_readonly_field_is_required_but_not_writable(self):
        fields = {"state": {"type": "selection", "required": True, "readonly": True}, "name": {"type": "char"}}

        schema = _build_bootstrap_schema(fields, 10)

        assert schema["required"] == ["state"]
        assert schema["writable"] == {"char": ["name"]}

    def test_a_field_without_a_type_is_grouped_as_unknown(self):
        assert _build_bootstrap_schema({"odd": {}}, 10)["writable"] == {"unknown": ["odd"]}


class TestSessionBootstrapBudget:
    def _bootstrap(self, monkeypatch, models: str) -> str:
        monkeypatch.setenv("MCP_BOOTSTRAP_MODELS", models)
        client = MagicMock()
        client.url, client.username = "https://budget.test", "budget"
        client.get_model_fields.side_effect = lambda model, attributes=None: _BIG_MODEL
        with patch.object(resources, "get_odoo_client", return_value=client):
            return read_resource("odoo://session-bootstrap")

    def test_the_default_five_models_fit_one_default_read(self, monkeypatch):
        out = self._bootstrap(monkeypatch, "res.partner,sale.order,account.move,product.product,stock.picking")

        data = json.loads(out)
        assert len(out) <= _READ_RESOURCE_MAX_CHARS
        assert "_truncated" not in data
        assert data["models_loaded"] == 5
        # 300-field models: the digest is a real cut, and the payload says where the rest is.
        assert all(0 < s["writable_shown"] < s["field_count"] for s in data["schemas"].values())
        assert "quick-schema" in data["note"]

    def test_the_field_budget_is_shared_between_the_models(self, monkeypatch):
        two = json.loads(self._bootstrap(monkeypatch, "res.partner,sale.order"))
        five = json.loads(
            self._bootstrap(monkeypatch, "res.partner,sale.order,account.move,product.product,stock.picking")
        )

        assert two["schemas"]["sale.order"]["writable_shown"] > five["schemas"]["sale.order"]["writable_shown"]

    def test_a_model_that_cannot_be_read_is_reported_beside_the_others(self, monkeypatch):
        monkeypatch.setenv("MCP_BOOTSTRAP_MODELS", "res.partner,x_missing.model")
        client = MagicMock()
        client.url, client.username = "https://budget.test", "budget-partial"

        def fields(model, attributes=None):
            if model == "x_missing.model":
                raise RuntimeError("no such model")
            return _BIG_MODEL

        client.get_model_fields.side_effect = fields
        with patch.object(resources, "get_odoo_client", return_value=client):
            data = json.loads(read_resource("odoo://session-bootstrap"))

        assert list(data["schemas"]) == ["res.partner"]
        assert list(data["errors"]) == ["x_missing.model"]

    def test_twenty_models_still_come_back_as_valid_json(self, monkeypatch):
        models = ",".join(f"x_custom.model_{i:02d}" for i in range(20))

        out = self._bootstrap(monkeypatch, models)

        assert len(out) <= _READ_RESOURCE_MAX_CHARS
        assert json.loads(out)["models_loaded"] == 20
