FROM python:3.12-slim

WORKDIR /app

# Create non-root user early for layer caching
RUN useradd --system --no-create-home --uid 1001 mcp

# Copy dependency manifests first for layer caching
COPY pyproject.toml README.md ./

# Resolve dependencies from pyproject.toml — never from a list duplicated here.
# A hand-copied list silently drifts from the declared constraints, which then
# never apply because the package itself is installed with --no-deps. It had
# already lost the deliberate `fastmcp<4` ceiling (carrying `>=3.2.0`) that
# tests/test_dependency_pins.py exists to guard, and it never listed
# `cryptography>=42` at all — token_crypto only imported because Authlib, a
# transitive FastMCP dependency, happens to pull cryptography in.
# Installing a stub package resolves the real dependency set into its own cached
# layer, so editing src/ does not re-resolve them; the stub is then replaced by
# the real package below.
RUN mkdir -p src/odoo_mcp \
    && touch src/odoo_mcp/__init__.py \
    && pip install --no-cache-dir . \
    && pip uninstall --yes odoo-mcp-19 \
    && rm -rf src

# Copy source and install the local package; dependencies are already resolved
# above, so --no-deps keeps this layer cheap without weakening any constraint.
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

USER mcp

EXPOSE 8080

# Default command - STDIO transport for Claude Desktop
CMD ["python", "-m", "odoo_mcp"]
