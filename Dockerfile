FROM python:3.12-slim

WORKDIR /app

# Create non-root user early for layer caching
RUN useradd --system --no-create-home --uid 1001 mcp

# Copy dependency manifests first for layer caching
COPY pyproject.toml README.md ./

# Install dependencies only (without local package)
RUN pip install --no-cache-dir \
    "fastmcp[tasks]>=3.2.0" \
    "requests>=2.32.4" \
    "python-dotenv>=1.0.0"

# Copy source and install local package without re-installing deps
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

USER mcp

EXPOSE 8080

# Default command - STDIO transport for Claude Desktop
CMD ["python", "-m", "odoo_mcp"]
