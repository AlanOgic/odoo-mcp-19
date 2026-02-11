"""
Odoo JSON-2 API client for Odoo 19+

This client uses the v2 JSON-2 API exclusively.
"""

import json
import os
import sys
import re
import urllib.parse
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from .arg_mapping import convert_args_to_v2


class OdooClient:
    """Client for interacting with Odoo 19+ via JSON-2 API"""

    def __init__(
        self,
        url: str,
        db: str,
        username: str,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
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

        # Setup session
        self.session = requests.Session()
        self.session.verify = verify_ssl
        self.session.headers['Content-Type'] = 'application/json'
        self.session.headers['X-Odoo-Database'] = db

        if api_key:
            self.session.headers['Authorization'] = f'Bearer {api_key}'

        # HTTP proxy support
        proxy = os.environ.get("HTTP_PROXY")
        if proxy:
            self.session.proxies = {"http": proxy, "https": proxy}

        # Parse hostname for logging
        parsed_url = urllib.parse.urlparse(self.url)
        self.hostname = parsed_url.netloc

        self._log_connection()

    def _log_connection(self):
        """Log connection details"""
        print(f"Connecting to Odoo 19+ at: {self.url}", file=sys.stderr)
        print(f"  Database: {self.db}", file=sys.stderr)
        print(f"  Auth: {'API Key' if self.api_key else 'Password'}", file=sys.stderr)

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

    def get_models(self) -> Dict[str, Any]:
        """Get all available models."""
        try:
            model_ids = self._execute("ir.model", "search", [])
            if not model_ids:
                return {"model_names": [], "models_details": {}}

            result = self._execute("ir.model", "read", model_ids, fields=["model", "name"])
            models = sorted([rec["model"] for rec in result])

            return {
                "model_names": models,
                "models_details": {
                    rec["model"]: {"name": rec.get("name", "")} for rec in result
                },
            }
        except Exception as e:
            print(f"Error retrieving models: {e}", file=sys.stderr)
            return {"model_names": [], "models_details": {}, "error": str(e)}

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
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

    def get_model_fields(self, model_name: str) -> Dict[str, Any]:
        """Get field definitions for a model."""
        try:
            return self._execute(model_name, "fields_get")
        except Exception as e:
            return {"error": str(e)}

    def get_model_doc(self, model_name: str) -> Optional[Dict[str, Any]]:
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
        except Exception:
            return None

    def search_read(
        self,
        model_name: str,
        domain: List,
        fields: Optional[List[str]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search and read records in one call."""
        try:
            kwargs = {}
            if fields is not None:
                kwargs["fields"] = fields
            if offset is not None:
                kwargs["offset"] = offset
            if limit is not None:
                kwargs["limit"] = limit
            if order is not None:
                kwargs["order"] = order

            return self._execute(model_name, "search_read", domain, **kwargs)
        except Exception as e:
            print(f"Error in search_read: {e}", file=sys.stderr)
            return []

    def read_records(
        self,
        model_name: str,
        ids: List[int],
        fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Read records by IDs."""
        try:
            kwargs = {}
            if fields is not None:
                kwargs["fields"] = fields
            return self._execute(model_name, "read", ids, **kwargs)
        except Exception as e:
            print(f"Error reading records: {e}", file=sys.stderr)
            return []


def load_config() -> Dict[str, str]:
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
            print(f"Loading config from: {env_path}", file=sys.stderr)
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
            print(f"Loading config from: {path}", file=sys.stderr)
            with open(path, "r") as f:
                return json.load(f)

    raise FileNotFoundError(
        "No Odoo configuration found.\n"
        "Set environment variables: ODOO_URL, ODOO_DB, ODOO_USERNAME, ODOO_API_KEY (or ODOO_PASSWORD)\n"
        "Or create a .env file or odoo_config.json"
    )


def get_odoo_client() -> OdooClient:
    """Get a configured Odoo client instance."""
    config = load_config()

    api_key = os.environ.get("ODOO_API_KEY")
    password = os.environ.get("ODOO_PASSWORD") or config.get("password")
    timeout = int(os.environ.get("ODOO_TIMEOUT", "30"))
    verify_ssl = os.environ.get("ODOO_VERIFY_SSL", "1").lower() in ["1", "true", "yes"]

    return OdooClient(
        url=config["url"],
        db=config["db"],
        username=config["username"],
        password=password,
        api_key=api_key,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )
