"""Tests for is_side_effect_method() predicate."""

import pytest

from odoo_mcp.safety import is_side_effect_method


@pytest.mark.parametrize(
    "method",
    [
        "create",
        "write",
        "unlink",
        "copy",
        "name_create",
        "load",
        "action_archive",
        "action_unarchive",
        "action_confirm",
        "action_post",
        "button_validate",
        "button_confirm",
        "button_cancel",
    ],
)
def test_known_side_effect_methods(method: str):
    assert is_side_effect_method(method) is True


@pytest.mark.parametrize(
    "method",
    [
        "search_read",
        "read",
        "search",
        "search_count",
        "fields_get",
        "name_search",
        "default_get",
        "has_access",
        "name_get",
    ],
)
def test_safe_methods_are_not_side_effects(method: str):
    assert is_side_effect_method(method) is False


@pytest.mark.parametrize(
    "method",
    [
        "action_my_custom_workflow",
        "button_do_something",
        "_action_private_hook",
    ],
)
def test_action_button_patterns(method: str):
    assert is_side_effect_method(method) is True


def test_empty_method_is_gated():
    # Fail-closed: an unrecognised method is never assumed to be a read.
    # (_validate_method rejects the empty string upstream anyway.)
    assert is_side_effect_method("") is True


def test_unknown_method_is_gated():
    # A method we have never heard of must be treated as a write. Assuming
    # otherwise is what let writes past the MCP_READ_ONLY kill-switch.
    assert is_side_effect_method("get_widget_count") is True


@pytest.mark.parametrize(
    "method",
    [
        # Documented in module_knowledge.json — all of these create or modify
        # records while matching neither a literal CRUD name nor action_* /
        # button_*, so a name-shape predicate waved them straight through.
        "add_members",
        "article_create",
        "article_duplicate",
        "channel_create",
        "convert_opportunity",
        "create_from_attachments",
        "create_from_binary_files",
        "create_from_urls",
        "document_create",
        "get_direct_response",
        "open_agent_chat",
        # Standard ORM write paths with read-shaped names.
        "message_post",
        "toggle_active",
    ],
)
def test_write_methods_without_action_prefix_are_gated(method: str):
    assert is_side_effect_method(method) is True
