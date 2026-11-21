"""Hints for HTTP errors the JSON-2 transport surfaces (utils.get_error_suggestion)."""

from odoo_mcp.utils import get_error_suggestion

# ir_http._auth_method_bearer answers this 401 when _check_credentials(scope='rpc') finds no key:
# on 19.0 a key whose scope is NULL or 'rpc' matches, from saas-19.4 (and 20.0) only scope 'rpc'
# (res_users._check_apikey_credentials: "AND scope = %(scope)s").
INVALID_KEY = (
    "Request failed: 401 UNAUTHORIZED for url: https://x/json/2/res.partner/search_read\n" "Odoo error: Invalid apikey"
)


def test_an_invalid_api_key_names_the_rpc_scope_requirement():
    hint = get_error_suggestion(INVALID_KEY, "res.partner", "search_read")

    assert "'rpc'" in hint
    assert "19.4" in hint
    assert "expired" in hint


def test_a_401_inside_another_error_does_not_get_the_api_key_hint():
    """'401' as a record id, an amount or part of a URL is not an authentication failure."""
    missing_record = (
        "Request failed: 404 NOT FOUND for url: https://x/json/2/res.partner/write\n"
        "Odoo error: Record does not exist or has been deleted. (Record: res.partner(401,), User: 2)"
    )

    hint = get_error_suggestion(missing_record, "res.partner", "write")

    assert "API key" not in (hint or "")
