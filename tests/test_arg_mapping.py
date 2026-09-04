"""Unit tests for positional → JSON-2 named-argument conversion.

These pin the contract from Odoo 19's External JSON-2 API
(``content/developer/reference/external_api.rst``):

  - Record-bound methods must place the recordset in the body ``ids`` key.
  - ``@api.model`` / search methods carry their own named params (``domain`` …).
  - All arguments are named; positional calling is not supported by JSON-2.

The regression these guard against: action/button methods (and otherwise
unmapped record methods) silently dropping their ``ids``, so the call would
run against an empty recordset.
"""

import pytest

from odoo_mcp.arg_mapping import convert_args_to_v2

# Methods that operate on a recordset: the leading [ids] arg must become "ids".
ACTION_METHODS = [
    "action_confirm",
    "action_cancel",
    "action_done",
    "action_draft",
    "action_validate",
    "action_post",
    "button_confirm",
    "button_cancel",
    "button_draft",
    "button_validate",
]


class TestActionMethodsCarryIds:
    """Action/button methods must forward the recordset as ``ids``."""

    def test_each_action_method_maps_leading_list_to_ids(self):
        for method in ACTION_METHODS:
            body = convert_args_to_v2(method, ([15],), {})
            assert body == {"ids": [15]}, f"{method} dropped its ids: {body!r}"

    def test_action_method_with_multiple_ids(self):
        body = convert_args_to_v2("button_validate", ([3, 7, 9],), {})
        assert body == {"ids": [3, 7, 9]}

    def test_action_method_preserves_context_kwarg(self):
        body = convert_args_to_v2("action_post", ([7],), {"context": {"lang": "fr_FR"}})
        assert body == {"ids": [7], "context": {"lang": "fr_FR"}}


class TestUnmappedRecordMethods:
    """Methods absent from the table but called on records must still send ids."""

    def test_unmapped_method_with_id_list_becomes_ids(self):
        # e.g. crm.lead action_set_won([9]) used by the lead_to_won workflow
        body = convert_args_to_v2("action_set_won", ([9],), {})
        assert body == {"ids": [9]}

    def test_unmapped_method_keeps_extra_kwargs(self):
        # convert_opportunity([9], partner_id=3) — ids must survive alongside kwargs
        body = convert_args_to_v2("convert_opportunity", ([9],), {"partner_id": 3})
        assert body == {"ids": [9], "partner_id": 3}

    def test_unmapped_method_with_no_args_is_empty(self):
        assert convert_args_to_v2("some_model_method", (), {}) == {}

    def test_unmapped_method_with_non_id_first_arg_is_not_coerced(self):
        # A leading string (not a recordset) must not be mislabeled as ids.
        body = convert_args_to_v2("name_create", ("Acme",), {})
        assert "ids" not in body


class TestExistingMappingsUnchanged:
    """Regression guard: the already-correct mappings keep working."""

    def test_read(self):
        assert convert_args_to_v2("read", ([1, 2],), {"fields": ["name"]}) == {
            "ids": [1, 2],
            "fields": ["name"],
        }

    def test_write(self):
        assert convert_args_to_v2("write", ([1], {"x": 1}), {}) == {
            "ids": [1],
            "vals": {"x": 1},
        }

    def test_unlink(self):
        assert convert_args_to_v2("unlink", ([1, 2],), {}) == {"ids": [1, 2]}

    def test_create_uses_vals_list(self):
        assert convert_args_to_v2("create", ([{"name": "X"}],), {}) == {"vals_list": [{"name": "X"}]}

    def test_search_maps_domain(self):
        assert convert_args_to_v2("search", ([["a", "=", 1]],), {}) == {"domain": [["a", "=", 1]]}

    def test_search_defaults_empty_domain(self):
        assert convert_args_to_v2("search_read", (), {"limit": 5}) == {
            "domain": [],
            "limit": 5,
        }

    def test_copy_sends_ids(self):
        # copy() runs on a recordset → the record selector belongs in ids.
        assert convert_args_to_v2("copy", ([5],), {}) == {"ids": [5]}


class TestFullPositionalMappings:
    """Positionals past the first must be forwarded, not silently dropped.

    Regression for the class of bug where ``search([], 0, 3)`` returned the
    server default page size and ``name_search('a', [['id','=',-1]])`` lost its
    domain and matched everything.
    """

    def test_search_forwards_offset_limit_order(self):
        assert convert_args_to_v2("search", ([], 0, 3, "name asc"), {}) == {
            "domain": [],
            "offset": 0,
            "limit": 3,
            "order": "name asc",
        }

    def test_search_read_forwards_fields_and_limit(self):
        assert convert_args_to_v2("search_read", ([], ["name"], 0, 5), {}) == {
            "domain": [],
            "fields": ["name"],
            "offset": 0,
            "limit": 5,
        }

    def test_search_count_forwards_limit(self):
        assert convert_args_to_v2("search_count", ([], 10), {}) == {"domain": [], "limit": 10}

    def test_read_forwards_fields_positionally(self):
        assert convert_args_to_v2("read", ([1], ["name"]), {}) == {"ids": [1], "fields": ["name"]}

    def test_name_search_keeps_its_domain(self):
        domain = [["id", "=", -1]]
        body = convert_args_to_v2("name_search", ("a", domain, "ilike", 5), {})
        assert body == {"name": "a", "domain": domain, "operator": "ilike", "limit": 5}

    def test_copy_forwards_default_overrides(self):
        assert convert_args_to_v2("copy", ([5], {"name": "Dup"}), {}) == {
            "ids": [5],
            "default": {"name": "Dup"},
        }

    def test_formatted_read_group_full_signature(self):
        body = convert_args_to_v2(
            "formatted_read_group",
            ([], ["state"], ["amount_total:sum"], [], 0, 10, "state"),
            {},
        )
        assert body == {
            "domain": [],
            "groupby": ["state"],
            "aggregates": ["amount_total:sum"],
            "having": [],
            "offset": 0,
            "limit": 10,
            "order": "state",
        }

    def test_fields_get_forwards_allfields_and_attributes(self):
        assert convert_args_to_v2("fields_get", (["name"], ["type"]), {}) == {
            "allfields": ["name"],
            "attributes": ["type"],
        }


class TestDefaultGetParameterName:
    """default_get's parameter is ``fields`` — the docstring's ``fields_list`` is stale."""

    def test_default_get_maps_to_fields(self):
        assert convert_args_to_v2("default_get", (["name"],), {}) == {"fields": ["name"]}


class TestReadGroupOrderbyExemption:
    """read_group really spells its sort parameter ``orderby``; it must not be renamed."""

    def test_read_group_positional_orderby_kept(self):
        body = convert_args_to_v2("read_group", ([], ["x"], ["y"], 0, None, "x desc", True), {})
        assert body["orderby"] == "x desc"
        assert "order" not in body

    def test_read_group_kwarg_orderby_not_renamed(self):
        body = convert_args_to_v2("read_group", ([], ["x"], ["y"]), {"orderby": "x desc"})
        assert body["orderby"] == "x desc"
        assert "order" not in body

    def test_search_kwarg_orderby_still_renamed_to_order(self):
        body = convert_args_to_v2("search", ([],), {"orderby": "name"})
        assert body == {"domain": [], "order": "name"}


class TestUnmappablePositionalRaises:
    """A positional with no JSON-2 name must fail loudly instead of being dropped."""

    def test_extra_positional_on_mapped_method_raises(self):
        with pytest.raises(ValueError) as exc:
            convert_args_to_v2("unlink", ([1], "surprise"), {})
        msg = str(exc.value)
        assert "unlink" in msg and "position 1" in msg and "kwargs_json" in msg

    def test_extra_positional_after_ids_fallback_raises(self):
        with pytest.raises(ValueError):
            convert_args_to_v2("action_set_won", ([9], {"stage": 1}), {})

    def test_non_list_positional_on_unmapped_method_raises(self):
        with pytest.raises(ValueError) as exc:
            convert_args_to_v2("some_model_method", ("x",), {})
        assert "no entry in V2_ARG_MAPPING" in str(exc.value)

    def test_ids_fallback_alone_does_not_raise(self):
        assert convert_args_to_v2("action_set_won", ([9],), {}) == {"ids": [9]}
