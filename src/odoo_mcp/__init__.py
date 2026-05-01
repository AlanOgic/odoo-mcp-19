"""
Odoo MCP Server for Odoo 19+

A Model Context Protocol server for interacting with Odoo 19+ via JSON-2 API.
"""

import logging as _logging
import sys as _sys

__version__ = "1.14.0"

# Configure the odoo_mcp logger namespace once: stderr, INFO+, bare format.
# Stderr is mandatory under STDIO MCP transport (stdout is reserved for protocol).
# Idempotent — only attaches a handler if none is configured yet, so downstream
# users can replace it via logging.getLogger("odoo_mcp").addHandler(...).
_pkg_logger = _logging.getLogger(__name__)
if not _pkg_logger.handlers:
    _h = _logging.StreamHandler(_sys.stderr)
    _h.setFormatter(_logging.Formatter("%(message)s"))
    _pkg_logger.addHandler(_h)
    _pkg_logger.setLevel(_logging.INFO)
    # Leave propagate=True so pytest's caplog and any root-level handlers also
    # receive our records. Hosts that want to silence us can call
    # logging.getLogger("odoo_mcp").handlers.clear() before importing sub-modules.
