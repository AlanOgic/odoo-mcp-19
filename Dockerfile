FROM python:3.12-slim

WORKDIR /app

# Copy all source files first
COPY pyproject.toml README.md ./
COPY src/ src/

# Install package
RUN pip install --no-cache-dir .

# Default command - STDIO transport for Claude Desktop
CMD ["python", "-m", "odoo_mcp"]
