"""
Odoo JSON-2 API client for Odoo 19+

This client uses the v2 JSON-2 API exclusively.
"""

import json
import logging
import os
import re
import threading
import urllib.parse
from typing import Any

import requests
from dotenv import load_dotenv

from .arg_mapping import convert_args_to_v2

logger = logging.getLogger(__name__)


class OdooClient:
    """Client for interacting with Odoo 19+ via JSON-2 API"""

    def __init__(
        self,
        url: str,
        db: str,
        username: str,
        password: str | None = None,
        api_key: str | None = None,
        timeout: int = 30,
        verify_ssl: bool = True,
    ):
        """
        Initialize the Odoo client.

        Args:
            url: Odoo server URL
            db: Database name
            username: Login username
            password: Login password (fallback if no API key)
            api_key: API key for authentication (recommended)
            timeout: Connection timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        # Ensure URL has a protocol
        if not re.match(r"^https?://", url):
            url = f"https://{url}"

        self.url = url.rstrip("/")
        self.db = db
        self.username = username
        self.password = password
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        # Priority: api_key > password
        self.auth_credential = api_key if api_key else password
        if not self.auth_credential:
            raise ValueError("Either api_key or password is required")

        # Parse hostname for logging and SaaS detection
        parsed_url = urllib.parse.urlparse(self.url)
        self.hostname = parsed_url.netloc

        # Setup session
        self.session = requests.Session()
        self.session.verify = verify_ssl
        self.session.headers['Content-Type'] = 'application/json'

        # Lift the default per-host connection pool from 10 to 20. Bundle and
        # session-bootstrap fan out up to 10 concurrent fetches; without this,
        # any concurrent execute_method call would block waiting for a free
        # connection.
        _adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.session.mount("https://", _adapter)
        self.session.mount("http://", _adapter)

        # Only set X-Odoo-Database for non-SaaS instances.
        # Odoo.com SaaS identifies the DB via subdomain — sending this header causes 404.
        hostname_lower = (parsed_url.hostname or "").lower()
        is_odoo_saas = hostname_lower.endswith(".odoo.com") or hostname_lower == "odoo.com"
        if db and not is_odoo_saas:
            self.session.headers['X-Odoo-Database'] = db

        if api_key:
            self.session.headers['Authorization'] = f'Bearer {api_key}'

        self._warn_insecure_ssl()

    def _warn_insecure_ssl(self):
        """Emit a security warning when SSL verification is disabled.

        Connection summary (URL / Database / Auth) lives in the verbose
        startup banner in `__main__._print_startup_banner`. This method now
        only carries the SSL-disabled MITM warning, which must fire whenever
        verification is off regardless of MCP_VERBOSE.
        """
        if not self.verify_ssl and self.url.startswith("https://"):
            logger.warning(
                "SSL verification is DISABLED. Connections are vulnerable to MITM attacks. "
                "Set ODOO_VERIFY_SSL=true for production."
            )

    def _execute(self, model: str, method: str, *args, **kwargs) -> Any:
        """
        Execute a method on an Odoo model using JSON-2 API.

        Args:
            model: Model name (e.g., 'res.partner')
            method: Method name (e.g., 'search_read')
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from Odoo
        """
        url = f"{self.url}/json/2/{model}/{method}"

        # Convert positional args to named args
        payload = convert_args_to_v2(method, tuple(args), kwargs)

        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)

            # Extract Odoo error details from response body BEFORE raise_for_status
            if response.status_code >= 400:
                try:
                    error_body = response.json()
                    odoo_error = error_body.get('message', '') or error_body.get('name', '')
                    odoo_debug = error_body.get('debug', '')
                    # Build informative error message
                    parts = [f"Request failed: {response.status_code} {response.reason} for url: {response.url}"]
                    if odoo_error:
                        parts.append(f"Odoo error: {odoo_error}")
                    if odoo_debug:
                        # Log full traceback server-side only — never forward to client
                        logger.debug("Server traceback for %s:\n%s", url, odoo_debug)
                    raise ValueError("\n".join(parts))
                except (json.JSONDecodeError, KeyError, AttributeError):
                    # If we can't parse the error body, fall back to raise_for_status
                    response.raise_for_status()

            result = response.json()

            # Handle wrapped response format
            if isinstance(result, dict) and 'result' in result:
                return result['result']
            return result

        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request timeout after {self.timeout}s: {e}")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Odoo: {e}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Request failed: {e}")

    def execute_method(self, model: str, method: str, *args, **kwargs) -> Any:
        """Execute an arbitrary method on a model."""
        return self._execute(model, method, *args, **kwargs)

    def get_models(self) -> dict[str, Any]:
        """Get all available models."""
        try:
            result = self._execute("ir.model", "search_read", [], fields=["model", "name"])
            if not result:
                return {"model_names": [], "models_details": {}}

            models = sorted([rec["model"] for rec in result])

            return {
                "model_names": models,
                "models_details": {
                    rec["model"]: {"name": rec.get("name", "")} for rec in result
                },
            }
        except Exception as e:
            logger.error("Error retrieving models: %s", e)
            return {"model_names": [], "models_details": {}, "error": str(e)}

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get information about a specific model."""
        try:
            result = self._execute(
                "ir.model",
                "search_read",
                [("model", "=", model_name)],
                fields=["name", "model"],
            )
            if not result:
                return {"error": f"Model {model_name} not found"}
            return result[0]
        except Exception as e:
            return {"error": str(e)}

    def get_model_fields(self, model_name: str) -> dict[str, Any]:
        """Get field definitions for a model."""
        try:
            return self._execute(model_name, "fields_get")
        except Exception as e:
            return {"error": str(e)}

    def get_model_doc(self, model_name: str) -> dict[str, Any] | None:
        """Fetch model documentation from /doc-bearer/<model>.json endpoint.

        The api_doc module is auto-installed in Odoo 19 (depends: ['web']).
        Requires the user to have the api_doc.group_allow_doc group.

        Returns the full model doc dict or None if unavailable.
        """
        url = f"{self.url}/doc-bearer/{model_name}.json"
        try:
            response = self.session.post(url, json={}, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            # JSON-2 may wrap in {"result": ...} or return directly
            if isinstance(data, dict) and "result" in data and isinstance(data["result"], dict):
                return data["result"]
            return data
        except Exception as e:
            logger.debug("get_model_doc failed for %s: %s: %s", model_name, type(e).__name__, e)
            return None

    def search_read(
        self,
        model_name: str,
        domain: list,
        fields: list[str] | None = None,
        offset: int | None = None,
        limit: int | None = None,
        order: str | None = None
    ) -> list[dict[str, Any]]:
        """Search and read records in one call."""
        kwargs: dict[str, Any] = {}
        if fields is not None:
            kwargs["fields"] = fields
        if offset is not None:
            kwargs["offset"] = offset
        if limit is not None:
            kwargs["limit"] = limit
        if order is not None:
            kwargs["order"] = order

        return self._execute(model_name, "search_read", domain, **kwargs)

    def read_records(
        self,
        model_name: str,
        ids: list[int],
        fields: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Read records by IDs."""
        kwargs: dict[str, Any] = {}
        if fields is not None:
            kwargs["fields"] = fields
        return self._execute(model_name, "read", ids, **kwargs)


def load_config() -> dict[str, str]:
    """Load Odoo configuration from environment or files."""
    # Try .env files
    env_paths = [
        ".env",
        os.path.expanduser("~/.config/odoo/.env"),
    ]

    custom_dir = os.environ.get("ODOO_CONFIG_DIR")
    if custom_dir:
        env_paths.insert(0, os.path.join(os.path.expanduser(custom_dir), ".env"))

    for env_path in env_paths:
        if os.path.exists(env_path):
            logger.info("Loading config from: %s", env_path)
            load_dotenv(dotenv_path=env_path, override=True)
            break

    # Check environment variables
    required = ["ODOO_URL", "ODOO_DB", "ODOO_USERNAME"]
    if all(var in os.environ for var in required):
        if os.environ.get("ODOO_API_KEY") or os.environ.get("ODOO_PASSWORD"):
            return {
                "url": os.environ["ODOO_URL"],
                "db": os.environ["ODOO_DB"],
                "username": os.environ["ODOO_USERNAME"],
                "password": os.environ.get("ODOO_PASSWORD", ""),
            }

    # Try JSON config files
    config_paths = [
        "./odoo_config.json",
        os.path.expanduser("~/.config/odoo/config.json"),
    ]

    for path in config_paths:
        if os.path.exists(path):
            logger.info("Loading config from: %s", path)
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in config file {path}: {e}") from e

    raise FileNotFoundError(
        "No Odoo configuration found.\n"
        "Set environment variables: ODOO_URL, ODOO_DB, ODOO_USERNAME, ODOO_API_KEY (or ODOO_PASSWORD)\n"
        "Or create a .env file or odoo_config.json"
    )


_client_lock = threading.Lock()
_client_instance: OdooClient | None = None


def get_odoo_client() -> OdooClient:
    """Get a configured Odoo client singleton instance."""
    global _client_instance
    if _client_instance is not None:
        return _client_instance
    with _client_lock:
        if _client_instance is not None:
            return _client_instance
        config = load_config()
        api_key = os.environ.get("ODOO_API_KEY")
        password = os.environ.get("ODOO_PASSWORD") or config.get("password")
        timeout = int(os.environ.get("ODOO_TIMEOUT", "30"))
        verify_ssl = os.environ.get("ODOO_VERIFY_SSL", "1").lower() in ["1", "true", "yes"]
        _client_instance = OdooClient(
            url=config["url"],
            db=config["db"],
            username=config["username"],
            password=password,
            api_key=api_key,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )
        return _client_instance
