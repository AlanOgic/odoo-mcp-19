"""Tests for is_side_effect_method() predicate."""
import pytest

from odoo_mcp.safety import is_side_effect_method


@pytest.mark.parametrize("method", [
    "create", "write", "unlink", "copy",
    "name_create", "load",
    "action_archive", "action_unarchive", "action_confirm", "action_post",
    "button_validate", "button_confirm", "button_cancel",
])
def test_known_side_effect_methods(method: str):
    assert is_side_effect_method(method) is True


@pytest.mark.parametrize("method", [
    "search_read", "read", "search", "search_count",
    "fields_get", "name_search", "default_get",
    "has_access", "name_get",
])
def test_safe_methods_are_not_side_effects(method: str):
    assert is_side_effect_method(method) is False


@pytest.mark.parametrize("method", [
    "action_my_custom_workflow",
    "button_do_something",
    "_action_private_hook",
])
def test_action_button_patterns(method: str):
    assert is_side_effect_method(method) is True


def test_empty_method_is_not_side_effect():
    assert is_side_effect_method("") is False


def test_unknown_read_like_method_is_not_side_effect():
    # A method we've never heard of that doesn't match any side-effect pattern.
    assert is_side_effect_method("get_widget_count") is False
