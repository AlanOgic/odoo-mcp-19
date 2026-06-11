"""
Cyanview workflow skill prompts for the Odoo MCP Server.

Serves the curated Cyanview business skills (quote, RMA, customer 360,
serial tracking, project design, shipping and inventory watchdogs) as MCP
prompt templates with typed arguments. In Claude Desktop they appear in
the "+" menu with their variables as input fields.

Skill bodies are markdown files in ``skills/`` (packaged via
``[tool.setuptools.package-data]``), copied from the curated
``~/.claude/skills/cyanview-*`` sources. Frontmatter is stripped at load.
Importing this module registers all prompts with the FastMCP instance.
"""

from pathlib import Path

from fastmcp.prompts import Message

from .app import mcp

_SKILLS_DIR = Path(__file__).parent / "skills"


def load_skill(name: str) -> str:
    """Load a skill markdown body, stripping the YAML frontmatter.

    Args:
        name: Skill file stem (e.g. "quote" for skills/quote.md).

    Returns:
        The skill instructions without frontmatter.

    Raises:
        FileNotFoundError: If the skill file does not exist.
    """
    text = (_SKILLS_DIR / f"{name}.md").read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + len("\n---"):]
    return text.strip()


def _skill_message(skill: str, request: str) -> list[Message]:
    """Build the single-message prompt: skill instructions + user request."""
    content = load_skill(skill)
    if request.strip():
        content += f"\n\n---\n\n**User request:** {request.strip()}"
    else:
        content += (
            "\n\n---\n\n**User request:** (none provided — "
            "ask the user what they need, then follow the workflow)"
        )
    return [Message(content)]


@mcp.prompt(name="cyanview-quote")
def cyanview_quote_prompt(request: str) -> list[Message]:
    """Build a Cyanview sales quotation in Odoo (products, pricing, SO)."""
    return _skill_message("quote", request)


@mcp.prompt(name="cyanview-rma")
def cyanview_rma_prompt(serial: str = "", issue: str = "") -> list[Message]:
    """Manage a Cyanview RMA/repair order in Odoo."""
    request = " — ".join(part for part in (serial.strip(), issue.strip()) if part)
    return _skill_message("rma", request)


@mcp.prompt(name="cyanview-customer-360")
def cyanview_customer_360_prompt(company: str) -> list[Message]:
    """Complete 360° briefing on a customer from Odoo (sales, RMA, CRM...)."""
    return _skill_message("customer_360", company)


@mcp.prompt(name="cyanview-serial-tracker")
def cyanview_serial_tracker_prompt(serial: str) -> list[Message]:
    """Trace a Cyanview device by serial number across its full lifecycle."""
    return _skill_message("serial_tracker", serial)


@mcp.prompt(name="cyanview-project-designer")
def cyanview_project_designer_prompt(requirements: str) -> list[Message]:
    """Design a Cyanview camera control architecture with a system diagram."""
    return _skill_message("project_designer", requirements)


@mcp.prompt(name="cyanview-shipping-watchdog")
def cyanview_shipping_watchdog_prompt(focus: str = "") -> list[Message]:
    """Audit confirmed sale orders that have not been fully shipped."""
    return _skill_message("shipping_watchdog", focus)


@mcp.prompt(name="cyanview-inventory-watchdog")
def cyanview_inventory_watchdog_prompt(focus: str = "") -> list[Message]:
    """Check stock levels, flag low inventory and suggest replenishment."""
    return _skill_message("inventory_watchdog", focus)
