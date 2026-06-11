"""Unit tests for the Cyanview skill prompts — no live Odoo needed.

Pins the skill-prompt layer: markdown files are packaged, frontmatter is
stripped, all seven prompts are registered with their typed arguments, and
rendering injects the user's variables.
"""

import asyncio

import pytest

from odoo_mcp.skill_prompts import _SKILLS_DIR, load_skill
from odoo_mcp.app import mcp

EXPECTED_SKILLS = [
    "quote",
    "rma",
    "customer_360",
    "serial_tracker",
    "project_designer",
    "shipping_watchdog",
    "inventory_watchdog",
]

EXPECTED_PROMPT_NAMES = {
    "cyanview-quote",
    "cyanview-rma",
    "cyanview-customer-360",
    "cyanview-serial-tracker",
    "cyanview-project-designer",
    "cyanview-shipping-watchdog",
    "cyanview-inventory-watchdog",
}


def test_all_skill_files_present():
    stems = {p.stem for p in _SKILLS_DIR.glob("*.md")}
    assert set(EXPECTED_SKILLS) <= stems


@pytest.mark.parametrize("name", EXPECTED_SKILLS)
def test_load_skill_strips_frontmatter(name):
    body = load_skill(name)
    assert body, f"skill {name} is empty"
    assert not body.startswith("---"), "frontmatter not stripped"


def test_load_skill_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_skill("does-not-exist")


def test_quote_skill_has_inlined_references():
    body = load_skill("quote")
    assert "Appendix A" in body
    assert "Appendix B" in body
    assert "references/product-catalog.md" not in body


def test_skill_prompts_registered():
    prompts = asyncio.run(mcp.list_prompts())
    names = {p.name for p in prompts}
    assert EXPECTED_PROMPT_NAMES <= names

    by_name = {p.name: p for p in prompts}
    quote_args = {a.name for a in (by_name["cyanview-quote"].arguments or [])}
    assert quote_args == {"request"}
    rma_args = {a.name for a in (by_name["cyanview-rma"].arguments or [])}
    assert rma_args == {"serial", "issue"}


def test_skill_prompt_renders_with_variables():
    result = asyncio.run(
        mcp.render_prompt("cyanview-serial-tracker", {"serial": "CY-RIO-15-042"})
    )
    text = result.messages[0].content.text
    assert "CY-RIO-15-042" in text
    assert "User request:" in text
