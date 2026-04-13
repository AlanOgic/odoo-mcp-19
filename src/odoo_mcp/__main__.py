"""
Entry point for running the Odoo MCP server
"""

import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any

from .server import mcp

DOCKER_IMAGE = "alanogik/odoo-mcp-19:latest"


def _prompt(label: str, default: str = "") -> str:
    """Prompt for input with optional default."""
    suffix = f" [{default}]" if default else ""
    value = input(f"  {label}{suffix}: ").strip()
    return value or default


def _prompt_secret(label: str) -> str:
    """Prompt for secret input (no echo)."""
    return getpass.getpass(f"  {label}: ").strip()


def _prompt_choice(label: str, options: list[str], default: str = "") -> str:
    """Prompt for a choice from a list."""
    opts = " / ".join(options)
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"  {label} ({opts}){suffix}: ").strip()
        if not value and default:
            return default
        if value in options:
            return value
        print(f"    Please choose: {opts}")


def _build_env_dict(config: dict) -> dict:
    """Build complete environment variable dictionary from config."""
    env = {
        "ODOO_URL": config["odoo_url"],
        "ODOO_DB": config["odoo_db"],
        "ODOO_USERNAME": config["odoo_username"],
        "MCP_SAFETY_MODE": config["safety_mode"],
    }
    if config["auth_method"] == "API Key":
        env["ODOO_API_KEY"] = config["auth_value"]
    else:
        env["ODOO_PASSWORD"] = config["auth_value"]

    env["MCP_TRANSPORT"] = config["transport"]
    if config["transport"] == "streamable-http":
        env["MCP_HOST"] = config["mcp_host"]
        env["MCP_PORT"] = config["mcp_port"]
        if config.get("mcp_api_key"):
            env["MCP_API_KEY"] = config["mcp_api_key"]

    return env


def _generate_env(config: dict) -> str:
    """Generate .env file content."""
    lines = ["# Odoo MCP Server Configuration"]
    for key, val in _build_env_dict(config).items():
        lines.append(f"{key}={val}")
    return "\n".join(lines) + "\n"


def _generate_docker_cmd(config: dict) -> str:
    """Generate docker run command using --env-file for security."""
    parts = ["docker run --rm -i"]

    if config["transport"] == "streamable-http":
        parts.append(f"-p {config['mcp_port']}:{config['mcp_port']}")

    parts.append("  --env-file .env")
    parts.append(f"  {DOCKER_IMAGE}")
    return " \\\n".join(parts)


def _generate_claude_desktop(config: dict) -> str:
    """Generate Claude Desktop JSON config snippet."""
    server: dict[str, Any] = {}
    if config["transport"] == "streamable-http":
        server = {
            "type": "streamable-http",
            "url": f"http://{config['mcp_host']}:{config['mcp_port']}/mcp",
        }
        if config.get("mcp_api_key"):
            server["headers"] = {
                "Authorization": f"Bearer {config['mcp_api_key']}"
            }
    else:
        server = {
            "command": "docker",
            "args": ["run", "--rm", "-i", DOCKER_IMAGE],
            "env": _build_env_dict(config),
        }

    return json.dumps({"mcpServers": {"odoo19": server}}, indent=2)


def run_setup_wizard():
    """Interactive setup wizard for Odoo MCP Server."""
    print()
    print("🔧 Odoo MCP Server — Setup Wizard")
    print()

    config = {}

    # --- Odoo Connection ---
    print("── Odoo Connection ──")
    config["odoo_url"] = _prompt("Odoo URL", "https://mycompany.odoo.com")
    config["odoo_db"] = _prompt("Database")
    config["odoo_username"] = _prompt("Username")
    config["auth_method"] = _prompt_choice(
        "Auth method", ["API Key", "Password"], "API Key"
    )
    config["auth_value"] = _prompt_secret(
        "API Key" if config["auth_method"] == "API Key" else "Password"
    )
    print()

    # --- MCP Transport ---
    print("── MCP Transport ──")
    config["transport"] = _prompt_choice(
        "Transport", ["stdio", "streamable-http"], "stdio"
    )
    if config["transport"] == "streamable-http":
        config["mcp_host"] = _prompt("Host", "0.0.0.0")
        config["mcp_port"] = _prompt("Port", "8080")
        print("  MCP API Key (bearer token) [Enter=auto-generate]: ", end="", flush=True)
        custom = getpass.getpass("").strip()
        if custom:
            config["mcp_api_key"] = custom
        else:
            import secrets

            config["mcp_api_key"] = secrets.token_urlsafe(32)
            print(f"  Generated: {config['mcp_api_key'][:8]}...")
    print()

    # --- Safety ---
    print("── Safety ──")
    config["safety_mode"] = _prompt_choice(
        "Safety mode", ["strict", "permissive"], "strict"
    )
    print()

    # --- Output ---
    print("── Output ──")
    print("  What to generate?")
    print("    [1] .env file")
    print("    [2] Docker run command")
    print("    [3] Claude Desktop JSON config")
    print("    [4] All of the above")
    choice = _prompt("Choice", "4")
    print()

    generate_env = choice in ("1", "4")
    generate_docker = choice in ("2", "4")
    generate_claude = choice in ("3", "4")

    # --- Generate outputs ---
    if generate_env:
        env_content = _generate_env(config)
        env_path = Path(".env")
        write = True
        if env_path.exists():
            overwrite = input("  .env already exists. Overwrite? (y/N): ").strip().lower()
            write = overwrite == "y"
        if write:
            env_path.write_text(env_content)
            print(f"  ✅ Written to {env_path.resolve()}")
        else:
            print("  ⏭  Skipped .env (not overwritten)")
            print()
            print(env_content)
        print()

    if generate_docker:
        print("── Docker Run Command ──")
        print()
        print(_generate_docker_cmd(config))
        print()

    if generate_claude:
        print("── Claude Desktop Config ──")
        print("  Add to ~/Library/Application Support/Claude/claude_desktop_config.json:")
        print()
        print(_generate_claude_desktop(config))
        print()

    print("✅ Setup complete!")


def _mask(value: str, keep: int = 4) -> str:
    """Mask a secret, keeping only the first `keep` chars."""
    if not value:
        return "(unset)"
    if len(value) <= keep:
        return "*" * len(value)
    return value[:keep] + "…" + "*" * max(0, len(value) - keep - 1)


def _print_startup_banner(transport: str, host: str, port: int) -> None:
    """Print a verbose startup banner to stderr.

    Stderr is used so STDIO transport can stay clean for the MCP protocol.
    Gated by MCP_VERBOSE — defaults to on for HTTP, off for STDIO.
    """
    try:
        from importlib.metadata import version as _pkg_version

        pkg_version = _pkg_version("odoo-mcp-19")
    except Exception:
        pkg_version = "unknown"

    odoo_url = os.environ.get("ODOO_URL", "(unset)")
    odoo_db = os.environ.get("ODOO_DB", "(unset)")
    odoo_user = os.environ.get("ODOO_USERNAME", "(unset)")
    odoo_key = os.environ.get("ODOO_API_KEY") or os.environ.get("ODOO_PASSWORD", "")
    odoo_timeout = os.environ.get("ODOO_TIMEOUT", "30")
    odoo_ssl = os.environ.get("ODOO_VERIFY_SSL", "true")

    safety_mode = os.environ.get("MCP_SAFETY_MODE", "strict")
    safety_audit = os.environ.get("MCP_SAFETY_AUDIT", "false")
    default_ctx = os.environ.get("MCP_DEFAULT_CONTEXT", "(none)")
    bootstrap = os.environ.get(
        "MCP_BOOTSTRAP_MODELS",
        "res.partner,sale.order,account.move,product.product,stock.picking",
    )

    line = "─" * 60
    out = sys.stderr
    print(file=out)
    print(r"   ____      __               __  _____________     _______", file=out)
    print(r"  / __ \____/ /___  ____     /  |/  / ____/ __ \   <  / __ \  __", file=out)
    print(r" / / / / __  / __ \/ __ \   / /|_/ / /   / /_/ /   / / /_/ /_/ /_", file=out)
    print(r"/ /_/ / /_/ / /_/ / /_/ /  / /  / / /___/ ____/   / /\__, /_  __/", file=out)
    print(r"\____/\__,_/\____/\____/  /_/  /_/\____/_/       /_//____/ /_/", file=out)
    print(file=out)
    print(f"  Odoo MCP Server  v{pkg_version}", file=out)
    print(line, file=out)
    print(f"  Transport     : {transport}", file=out)
    if transport == "streamable-http":
        print(f"  Bind          : http://{host}:{port}", file=out)
        print(
            f"  Auth          : Bearer (MCP_API_KEY={_mask(os.environ.get('MCP_API_KEY', ''))})",
            file=out,
        )
    print(file=out)
    print("  ── Odoo connection ──", file=out)
    print(f"  URL           : {odoo_url}", file=out)
    print(f"  Database      : {odoo_db}", file=out)
    print(f"  User          : {odoo_user}", file=out)
    print(f"  Credential    : {_mask(odoo_key)}", file=out)
    print(f"  Timeout       : {odoo_timeout}s", file=out)
    print(f"  Verify SSL    : {odoo_ssl}", file=out)
    if odoo_ssl.lower() in ("0", "false", "no", "off") and odoo_url.startswith("https://"):
        print(
            "  ⚠ WARNING    : SSL verification is DISABLED — vulnerable to MITM",
            file=out,
        )
    print(file=out)
    print("  ── Safety layer ──", file=out)
    print(f"  Mode          : {safety_mode}", file=out)
    print(f"  Audit log     : {safety_audit}", file=out)
    print(file=out)
    print("  ── DX defaults ──", file=out)
    print(f"  Default ctx   : {default_ctx}", file=out)
    print(f"  Bootstrap     : {bootstrap}", file=out)
    print(file=out)
    print("  ── Capabilities ──", file=out)
    print("  Tools         : 5  (execute_method, batch_execute, execute_workflow,", file=out)
    print("                      configure_odoo, read_resource)", file=out)
    print("  Resources     : 27 (odoo:// schemas, workflows, bundles, methods, ...)", file=out)
    print("  Prompts       : 13 (quote-to-cash, customer-360, daily-operations, ...)", file=out)
    print(line, file=out)
    out.flush()


def main():
    """Main entry point for the MCP server"""
    if "--setup" in sys.argv:
        run_setup_wizard()
        return

    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8080"))

    # Verbose startup banner. On by default for both transports — the banner
    # writes only to stderr, so STDIO transport stays clean (stdout is reserved
    # for the MCP protocol). Set MCP_VERBOSE=false to silence.
    # The banner is delayed so it lands AFTER FastMCP's own "Starting MCP
    # server" log line rather than racing it.
    if os.environ.get("MCP_VERBOSE", "true").lower() in ("1", "true", "yes", "on"):
        import threading

        threading.Timer(
            0.4, _print_startup_banner, args=(transport, host, port)
        ).start()

    # Log auth status for HTTP transport
    if transport == "streamable-http":
        if not os.environ.get("MCP_API_KEY"):
            print(
                "ERROR: MCP_API_KEY must be set when using streamable-http transport.\n"
                "Set MCP_API_KEY environment variable or run --setup to configure.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("🔐 Authentication enabled (Bearer token required)", file=sys.stderr)
        mcp.run(transport="streamable-http", host=host, port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
