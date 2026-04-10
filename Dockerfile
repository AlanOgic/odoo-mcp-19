FROM python:3.12-slim

WORKDIR /app

# Copy all source files first
COPY pyproject.toml README.md ./
COPY src/ src/

# Install package and create non-root user
RUN pip install --no-cache-dir . && \
    useradd --system --no-create-home --uid 1001 mcp

USER mcp

# Default command - STDIO transport for Claude Desktop
CMD ["python", "-m", "odoo_mcp"]
