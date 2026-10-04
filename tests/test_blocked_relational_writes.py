"""BLOCKED_MODELS must hold against writes that reach a blocked model indirectly.

The classifier used to look only at the top-level model name, so a blocked model
was reachable through x2many commands on an allowed one (``res.partner.user_ids``
creates or edits ``res.users``) and through unlisted proxy models (password
wizards, server actions, settings wizards).
"""

import pytest

from odoo_mcp.safety import BLOCKED_MODELS, SAFE_METHODS, RiskLevel, classify_operation

SCHEMAS = {
    "res.partner": {
        "name": {"type": "char"},
        "user_id": {"type": "many2one", "relation": "res.users"},
        "user_ids": {"type": "one2many", "relation": "res.users"},
        "child_ids": {"type": "one2many", "relation": "res.partner"},
        "category_id": {"type": "many2many", "relation": "res.partner.category"},
    },
    "res.company": {"user_ids": {"type": "many2many", "relation": "res.users"}},
    "sale.order": {"order_line": {"type": "one2many", "relation": "sale.order.line"}},
    "sale.order.line": {"name": {"type": "char"}},
    "res.partner.category": {"name": {"type": "char"}},
    "ir.model": {
        "name": {"type": "char"},
        "rule_ids": {"type": "one2many", "relation": "ir.rule"},
        "access_ids": {"type": "one2many", "relation": "ir.model.access"},
    },
}


class Loader:
    def __init__(self, schemas=SCHEMAS):
        self.schemas = schemas
        self.calls = []

    def __call__(self, model):
        self.calls.append(model)
        return self.schemas.get(model, {})


def _classify(model, method, args=None, kwargs=None, loader=None):
    return classify_operation(model, method, args, kwargs, fields_loader=loader or Loader())


class TestX2manyCommandsOnBlockedComodel:
    @pytest.mark.parametrize(
        "command",
        [
            [0, 0, {"login": "x", "name": "x"}],
            [1, 5, {"password": "x"}],
            [2, 5, 0],
            [3, 5, 0],
            [5, 0, 0],
            [6, 0, []],
        ],
    )
    def test_one2many_command_on_blocked_comodel_is_blocked(self, command):
        result = _classify("res.partner", "write", [[7], {"user_ids": [command]}])

        assert result.risk_level == RiskLevel.BLOCKED
        assert result.requires_confirmation is False
        assert "res.users" in result.blocked_reason
        assert "user_ids" in result.blocked_reason

    @pytest.mark.parametrize("code", [True, 1.0, 0.0, False, 2.0])
    def test_command_code_equal_to_a_write_code_is_blocked(self, code):
        """Odoo compares command codes with ==, so True and 1.0 are UPDATE."""
        result = _classify("res.partner", "write", [[7], {"user_ids": [[code, 5, {"password": "x"}]]}])

        assert result.risk_level == RiskLevel.BLOCKED

    @pytest.mark.parametrize("value", [[5], [], [5, 6], [{"login": "x"}], ["x"], [[[1], 5, {}]]])
    def test_any_other_list_on_a_one2many_to_a_blocked_comodel_is_blocked(self, value):
        """A bare id list is an implicit SET; anything unrecognised is refused."""
        result = _classify("res.partner", "write", [[7], {"user_ids": value}])

        assert result.risk_level == RiskLevel.BLOCKED

    def test_context_defaults_are_inspected(self):
        context = {"lang": "fr_FR", "default_user_ids": [[0, 0, {"login": "x"}]]}

        result = _classify("res.partner", "create", [[{"name": "x"}]], {"context": context})

        assert result.risk_level == RiskLevel.BLOCKED

    @pytest.mark.parametrize(
        "columns",
        [["name", "user_ids/login"], ["name", "user_ids"], ["child_ids/user_ids/login"], ["user_ids/id"]],
    )
    def test_load_columns_reaching_a_blocked_comodel_are_blocked(self, columns):
        rows = [["x"] * len(columns)]

        assert _classify("res.partner", "load", [columns, rows]).risk_level == RiskLevel.BLOCKED
        assert _classify("res.partner", "load", kwargs={"fields": columns, "data": rows}).risk_level == (
            RiskLevel.BLOCKED
        )

    def test_load_columns_on_allowed_comodels_pass(self):
        columns = ["name", "user_id/id", "category_id/id", "child_ids/name"]

        assert _classify("res.partner", "load", [columns, [["a", "b", "c", "d"]]]).risk_level == RiskLevel.MEDIUM

    def test_vals_passed_by_name_are_inspected(self):
        result = _classify("res.partner", "write", kwargs={"ids": [7], "vals": {"user_ids": [[1, 5, {"x": 1}]]}})

        assert result.risk_level == RiskLevel.BLOCKED

    def test_every_record_of_a_create_is_inspected(self):
        vals_list = [{"name": "ok"}, {"name": "bad", "user_ids": [[0, 0, {"login": "x"}]]}]

        assert _classify("res.partner", "create", [vals_list]).risk_level == RiskLevel.BLOCKED
        assert _classify("res.partner", "create", kwargs={"vals_list": vals_list}).risk_level == RiskLevel.BLOCKED

    def test_nested_commands_are_followed_through_allowed_comodels(self):
        vals = {"child_ids": [[0, 0, {"name": "kid", "user_ids": [[0, 0, {"login": "x"}]]}]]}

        assert _classify("res.partner", "write", [[7], vals]).risk_level == RiskLevel.BLOCKED

    def test_unlisted_write_method_carrying_vals_is_inspected(self):
        result = _classify("res.partner", "web_save", kwargs={"ids": [7], "vals": {"user_ids": [[1, 5, {"x": 1}]]}})

        assert result.risk_level == RiskLevel.BLOCKED

    @pytest.mark.parametrize("command", [[1, 5, {"name": "x"}], [0, 0, {"login": "x"}], [2, 5, 0]])
    def test_many2many_write_commands_on_blocked_comodel_are_blocked(self, command):
        result = _classify("res.company", "write", [[1], {"user_ids": [command]}])

        assert result.risk_level == RiskLevel.BLOCKED

    @pytest.mark.parametrize("value", [[[4, 5, 0]], [[3, 5, 0]], [[6, 0, [5, 6]]], [5, 6], []])
    def test_many2many_link_commands_on_blocked_comodel_are_allowed(self, value):
        result = _classify("res.company", "write", [[1], {"user_ids": value}])

        assert result.risk_level == RiskLevel.MEDIUM


class TestClearingAnX2manyWithNullOrFalse:
    """Odoo reads ``None``/``False`` on an x2many as ``[Command.clear()]``.

    On a one2many whose inverse is ``ondelete="cascade"`` that deletes every line
    (``odoo/orm/fields_relational.py``), so the value must get the same verdict as
    an explicit ``[[5, 0, 0]]``.
    """

    @pytest.mark.parametrize("value", [None, False])
    def test_clearing_a_one2many_to_a_blocked_comodel_is_blocked(self, value):
        result = _classify("res.partner", "write", [[7], {"user_ids": value}])

        assert result.risk_level == RiskLevel.BLOCKED
        assert "res.users" in result.blocked_reason

    @pytest.mark.parametrize("field_name, comodel", [("rule_ids", "ir.rule"), ("access_ids", "ir.model.access")])
    def test_clearing_the_rules_of_a_model_through_ir_model_is_blocked(self, field_name, comodel):
        result = _classify("ir.model", "write", [[42], {field_name: None}])

        assert result.risk_level == RiskLevel.BLOCKED
        assert comodel in result.blocked_reason

    def test_clearing_inside_an_update_command_is_followed(self):
        vals = {"child_ids": [[1, 8, {"user_ids": False}]]}

        assert _classify("res.partner", "write", [[7], vals]).risk_level == RiskLevel.BLOCKED

    def test_clearing_through_a_named_vals_argument_is_blocked(self):
        result = _classify("res.partner", "write", kwargs={"ids": [7], "vals": {"user_ids": None}})

        assert result.risk_level == RiskLevel.BLOCKED

    @pytest.mark.parametrize("value", [None, False])
    def test_clearing_a_many2many_to_a_blocked_comodel_is_allowed_like_command_5(self, value):
        """Detaching every user from a company is the same as ``[[5, 0, 0]]``: link-only."""
        result = _classify("res.company", "write", [[1], {"user_ids": value}])

        assert result.risk_level == RiskLevel.MEDIUM

    def test_null_and_false_on_scalar_and_many2one_fields_pass(self):
        result = _classify("res.partner", "write", [[7], {"name": False, "user_id": None}])

        assert result.risk_level == RiskLevel.MEDIUM

    def test_clearing_inside_a_create_command_is_followed(self):
        vals = {"child_ids": [[0, 0, {"name": "kid", "user_ids": None}]]}

        assert _classify("res.partner", "write", [[7], vals]).risk_level == RiskLevel.BLOCKED

    def test_clearing_through_a_context_default_is_blocked(self):
        result = _classify("res.partner", "create", [[{"name": "x"}]], {"context": {"default_user_ids": None}})

        assert result.risk_level == RiskLevel.BLOCKED

    @pytest.mark.parametrize("value", [0, ""])
    def test_other_falsy_values_are_not_a_clear(self, value):
        """Odoo uses identity checks: only None and False become Command.clear()."""
        assert _classify("res.partner", "write", [[7], {"user_ids": value}]).risk_level == RiskLevel.MEDIUM

    def test_unavailable_schema_blocks_a_payload_clearing_a_field(self):
        result = _classify("res.partner", "write", [[7], {"user_ids": None}], loader=Loader({}))

        assert result.risk_level == RiskLevel.BLOCKED
        assert "schema" in result.blocked_reason.lower()


class TestDictShorthandInX2manyLists:
    """Odoo's ``convert_to_cache`` reads a bare dict in an x2many list as a new record
    (``comodel.new(vals)``), and ``default_get`` sends every ``default_<field>``
    context key through it, so the dict is created on ``create``."""

    def test_a_dict_in_a_context_default_is_followed_to_a_blocked_comodel(self):
        context = {"default_child_ids": [{"name": "kid", "user_ids": [[0, 0, {"login": "x"}]]}]}

        result = _classify("res.partner", "create", [[{"name": "x"}]], {"context": context})

        assert result.risk_level == RiskLevel.BLOCKED
        assert "res.users" in result.blocked_reason

    def test_a_dict_in_vals_is_followed_to_a_blocked_comodel(self):
        vals = {"child_ids": [{"name": "kid", "user_ids": [[0, 0, {"login": "x"}]]}]}

        assert _classify("res.partner", "write", [[7], vals]).risk_level == RiskLevel.BLOCKED

    def test_a_dict_on_an_allowed_comodel_passes(self):
        context = {"default_child_ids": [{"name": "kid"}]}

        result = _classify("res.partner", "create", [[{"name": "x"}]], {"context": context})

        assert result.risk_level == RiskLevel.MEDIUM


class TestLegitimateWritesStillPass:
    def test_many2one_to_blocked_model_is_not_a_write_on_it(self):
        assert _classify("res.partner", "write", [[7], {"user_id": 5}]).risk_level == RiskLevel.MEDIUM

    def test_commands_on_allowed_comodel_pass(self):
        result = _classify("sale.order", "create", [[{"order_line": [[0, 0, {"name": "line"}]]}]])

        assert result.risk_level == RiskLevel.MEDIUM

    def test_schema_is_not_fetched_when_payload_has_no_list_value(self):
        loader = Loader()

        _classify("res.partner", "write", [[7], {"name": "x", "user_id": 5}], loader=loader)

        assert loader.calls == []

    def test_list_value_on_a_non_relational_field_passes(self):
        result = _classify("res.partner", "write", [[7], {"name": [[1, 5, {"x": 1}]]}])

        assert result.risk_level == RiskLevel.MEDIUM

    def test_id_list_on_allowed_comodel_passes(self):
        assert _classify("res.partner", "write", [[7], {"category_id": [1, 2]}]).risk_level == RiskLevel.MEDIUM

    def test_safe_method_never_fetches_schema(self):
        loader = Loader()

        result = _classify("res.partner", "search_read", kwargs={"domain": [["id", "in", [[1, 2, 3]]]]}, loader=loader)

        assert result.risk_level == RiskLevel.SAFE
        assert loader.calls == []

    def test_without_a_loader_classification_is_unchanged(self):
        result = classify_operation("res.partner", "write", [[7], {"user_ids": [[1, 5, {"x": 1}]]}])

        assert result.risk_level == RiskLevel.MEDIUM


class TestFailClosed:
    def test_unavailable_schema_blocks_a_payload_carrying_commands(self):
        result = _classify("res.partner", "write", [[7], {"user_ids": [[1, 5, {"x": 1}]]}], loader=Loader({}))

        assert result.risk_level == RiskLevel.BLOCKED
        assert "schema" in result.blocked_reason.lower()

    def test_unavailable_comodel_schema_blocks_nested_commands(self):
        schemas = {"res.partner": SCHEMAS["res.partner"]}
        vals = {"category_id": [[0, 0, {"child_ids": [[0, 0, {"name": "x"}]]}]]}

        result = _classify("res.partner", "write", [[7], vals], loader=Loader(schemas))

        assert result.risk_level == RiskLevel.BLOCKED


class TestProxyModelsAreBlocked:
    @pytest.mark.parametrize(
        "model",
        [
            "ir.access",
            "ir.model.data",
            "res.groups.privilege",
            "res.config.settings",
            "change.password.wizard",
            "change.password.user",
            "change.password.own",
            "base_import.import",
            "base.module.upgrade",
            "base.module.uninstall",
            "base.module.update",
            "base.module.install.request",
            "ir.mail_server",
            "fetchmail.server",
            "auth.oauth.provider",
            "res.company.ldap",
            "portal.wizard",
            "portal.wizard.user",
            "res.users.apikeys.description",
            "res.users.identitycheck",
            "res.users.deletion",
            "res.device",
            "res.session",
            "auth_totp.wizard",
            "auth_totp.device",
            "auth.passkey.key",
            "auth.passkey.key.create",
        ],
    )
    def test_proxy_model_is_blocked_for_writes_and_actions(self, model):
        assert model in BLOCKED_MODELS
        assert classify_operation(model, "create", [[{}]]).risk_level == RiskLevel.BLOCKED
        assert classify_operation(model, "run", [[1]]).risk_level == RiskLevel.BLOCKED

    def test_legacy_access_models_stay_blocked(self):
        assert {"ir.rule", "ir.model.access"} <= BLOCKED_MODELS


class TestSafeMethodsAreReads:
    def test_default_get_is_not_safe(self):
        assert "default_get" not in SAFE_METHODS
        assert classify_operation("sale.order.line", "default_get", role="readonly").risk_level == RiskLevel.BLOCKED

    def test_name_get_is_not_safe(self):
        assert "name_get" not in SAFE_METHODS


class TestSafetyModeTypoFailsClosed:
    @pytest.mark.parametrize("raw", ["stirct", "strict ", "true", ""])
    def test_unrecognised_mode_still_gates_medium_writes(self, monkeypatch, raw):
        monkeypatch.setenv("MCP_SAFETY_MODE", raw)

        result = classify_operation("res.partner", "write", [[7], {"name": "x"}])

        assert result.requires_confirmation is True

    def test_padded_locked_value_still_enforces_the_allowlist(self, monkeypatch):
        monkeypatch.setenv("MCP_SAFETY_MODE", " LOCKED")

        result = classify_operation("res.partner", "write", [[7], {"name": "x"}])

        assert result.risk_level == RiskLevel.BLOCKED

    def test_permissive_is_still_honoured(self, monkeypatch):
        monkeypatch.setenv("MCP_SAFETY_MODE", "Permissive")

        result = classify_operation("res.partner", "write", [[7], {"name": "x"}])

        assert result.requires_confirmation is False


class TestPrivilegedModels:
    """Automation, scheduling and schema models run code or reshape the database.

    The operator's own agent (stdio, env-admin, ``admin`` role) must be able to
    build automations and Studio-style fields, so they are not blocked — but every
    side-effect call on them confirms, in every mode, and any other role is refused.
    """

    MODELS = [
        "ir.actions.server",
        "base.automation",
        "ir.cron",
        "ir.model",
        "ir.model.fields",
        "ir.default",
        "ir.ui.view",
    ]

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("role", [None, "admin"])
    @pytest.mark.parametrize("mode", ["strict", "permissive"])
    @pytest.mark.parametrize("method", ["create", "write", "run", "method_direct_trigger", "unlink"])
    def test_admin_side_effects_always_confirm(self, monkeypatch, model, role, mode, method):
        monkeypatch.setenv("MCP_SAFETY_MODE", mode)

        result = classify_operation(model, method, [[1]], role=role)

        assert result.risk_level == RiskLevel.HIGH
        assert result.requires_confirmation is True

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("method", ["create", "write", "run", "unlink"])
    def test_other_roles_are_refused(self, model, method):
        result = classify_operation(model, method, [[1]], role="support")

        assert result.risk_level == RiskLevel.BLOCKED
        assert "admin" in result.blocked_reason.lower()

    @pytest.mark.parametrize("model", MODELS)
    def test_not_in_blocked_models(self, model):
        assert model not in BLOCKED_MODELS

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("role", [None, "admin", "support", "readonly"])
    def test_reads_stay_open_for_every_role(self, model, role):
        assert classify_operation(model, "search_read", role=role).risk_level == RiskLevel.SAFE

    def test_other_roles_cannot_reach_them_through_x2many_commands(self):
        schemas = {"some.model": {"action_ids": {"type": "one2many", "relation": "ir.actions.server"}}}
        vals = {"action_ids": [[0, 0, {"state": "code", "code": "pass"}]]}

        blocked = classify_operation("some.model", "write", [[1], vals], role="support", fields_loader=Loader(schemas))
        allowed = classify_operation("some.model", "write", [[1], vals], role="admin", fields_loader=Loader(schemas))

        assert blocked.risk_level == RiskLevel.BLOCKED
        assert allowed.risk_level == RiskLevel.MEDIUM


def test_reads_on_blocked_models_stay_open():
    """The connected account's Odoo rights decide what is readable, not this server."""
    for model in ("ir.config_parameter", "res.users", "ir.rule", "ir.mail_server"):
        assert classify_operation(model, "search_read").risk_level == RiskLevel.SAFE
