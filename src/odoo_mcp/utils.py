"""
Utility functions for the Odoo MCP Server.

Functions used by resources and tools for error handling, documentation,
schema building, and issue tracking.
"""

import json
import logging
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, cast

from .api_reference import build_api_index
from .constants import (
    _API_INDEX_CACHE,
    _API_INDEX_CACHE_LOCK,
    _API_INDEX_CACHE_MAX_ENTRIES,
    _API_INDEX_REFUSALS,
    _DOC_CACHE,
    _DOC_CACHE_LOCK,
    _DOC_CACHE_MAX_ENTRIES,
    _DOC_CACHE_TTL,
    _DOC_NEGATIVE_TTL,
    _DOC_REFUSALS,
    _RUNTIME_ISSUES_LOCK,
    ERROR_CATEGORIES,
    MODULE_KNOWLEDGE,
    RUNTIME_MODEL_ISSUES,
)
from .odoo_client import get_odoo_client

logger = logging.getLogger(__name__)

# ----- Compact Schema Builder -----


def _build_compact_schema(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Build an ultra-compact schema representation from fields_get output.

    Returns a dict with:
    - fields: {name: {t, req?, ro?, rel?, sel?}} with short keys
    - required_fields: list of required field names (top-level for quick access)

    Typically 60-80% smaller than /fields (absolute size scales with the
    model's field count).
    """
    compact = {}
    required = []
    for name, meta in fields.items():
        ftype = meta.get("type", "")
        entry: Dict[str, Any] = {"t": ftype}
        if meta.get("required"):
            entry["req"] = True
            required.append(name)
        if meta.get("readonly"):
            entry["ro"] = True
        if ftype in ("many2one", "one2many", "many2many"):
            entry["rel"] = meta.get("relation", "")
        # Only include selection values for 'state' field (high-value, small)
        if ftype == "selection" and name == "state" and meta.get("selection"):
            entry["sel"] = meta["selection"]
        compact[name] = entry
    return {"fields": compact, "required_fields": required}


def _strip_html(html_str: str) -> str:
    """Strip HTML tags and normalize whitespace for plain text display."""
    if not html_str:
        return ""
    text = re.sub(r"<[^>]+>", "", html_str)
    return " ".join(text.split()).strip()


def _client_identity(client: Any) -> tuple[Any, Any]:
    """``(url, username)`` of an OdooClient, the identity part of a cache key.

    Odoo filters fields, docs and the api index by the caller's rights, so cached
    copies must never cross identities.
    """
    return (getattr(client, "url", None), getattr(client, "username", None))


def _dumps(payload: Any) -> str:
    """Serialize a resource payload without whitespace.

    The reader is an agent: indentation costs it tokens and eats into the
    15 000-char ``read_resource`` budget (a 150-field ``/fields`` went from
    truncated to whole). ``tests/test_resource_payloads.py`` checks every route.
    """
    return json.dumps(payload, separators=(",", ":"))


def _cached_doc_answer(
    cache: Dict[Any, tuple[float, dict]], refusals: Dict[Any, float], key: Any, now: float
) -> tuple[bool, Optional[dict]]:
    """``(True, doc)`` for a fresh document, ``(True, None)`` for a recent refusal, else ``(False, None)``.

    The caller holds the lock guarding both dicts.
    """
    entry = cache.get(key)
    if entry is not None and now - entry[0] < _DOC_CACHE_TTL:
        return True, entry[1]
    refused_at = refusals.get(key)
    if refused_at is not None and now - refused_at < _DOC_NEGATIVE_TTL:
        return True, None
    return False, None


def _store_doc_answer(
    cache: Dict[Any, tuple[float, dict]],
    refusals: Dict[Any, float],
    key: Any,
    now: float,
    doc: Optional[dict],
    cap: int,
) -> None:
    """Record a document, or a refusal when ``doc`` is None, each dict under its own ``cap`` (oldest evicted).

    The caller holds the lock guarding both dicts.
    """
    if doc is None:
        refusals[key] = now
        while len(refusals) > cap:
            del refusals[min(refusals, key=refusals.__getitem__)]
        return
    cache[key] = (now, doc)
    refusals.pop(key, None)
    while len(cache) > cap:
        del cache[min(cache, key=lambda k: cache[k][0])]


def get_server_version() -> Optional[tuple[int, int]]:
    """The caller's Odoo (major, minor), or None when no client can be built or the version is unreadable.

    Same identity rule as ``_get_live_doc``: a ``PermissionError`` propagates.
    """
    try:
        odoo = get_odoo_client()
    except PermissionError:
        raise
    except Exception as e:
        logger.warning("Odoo server version unavailable: %s", e)
        return None
    return odoo.server_version()


def _get_live_doc(model_name: str) -> Optional[Dict[str, Any]]:
    """Fetch live model docs from /doc-bearer/ with in-memory caching.

    Cached per caller identity: Odoo builds the document for the requesting user.
    Returns the doc dict if available, or None when the endpoint fails -- the
    caller then falls back to static data. Odoo's refusal (403/404), or a 200 that
    carries no document, is cached for ``_DOC_NEGATIVE_TTL``, so an API user without
    ``api_doc.group_allow_doc`` does not pay a failing request per call; a transient
    failure is not, the next call retries.
    A caller whose identity cannot be resolved (``PermissionError``) is refused, not
    answered from static data.
    """
    try:
        odoo = get_odoo_client()
    except PermissionError:
        raise
    except Exception as e:
        logger.warning("/doc-bearer/ unavailable for %s: %s", model_name, e)
        return None

    key = (*_client_identity(odoo), model_name)
    now = time.time()
    with _DOC_CACHE_LOCK:
        hit, cached = _cached_doc_answer(_DOC_CACHE, _DOC_REFUSALS, key, now)
    if hit:
        return cached

    try:
        fetched = odoo.get_model_doc(model_name)
    except Exception as e:
        logger.warning("/doc-bearer/ unavailable for %s: %s", model_name, e)
        return None
    doc = fetched if isinstance(fetched, dict) and "methods" in fetched else None
    with _DOC_CACHE_LOCK:
        _store_doc_answer(_DOC_CACHE, _DOC_REFUSALS, key, now, doc, _DOC_CACHE_MAX_ENTRIES)
    return doc


def get_compact_api_index() -> Optional[Dict[str, Any]]:
    """``build_api_index`` of the current identity's ``/doc-bearer/index.json``, cached.

    Keyed by (url, username) because the index is filtered by the caller's
    groups — in multi-user mode two registry users must not share it. Only the
    compact form is cached: nothing serves the raw ~2 MB index. Same TTLs and
    refusal handling as ``_get_live_doc``; at most ``_API_INDEX_CACHE_MAX_ENTRIES``
    identities (oldest evicted).
    """
    client = get_odoo_client()
    key = _client_identity(client)
    now = time.time()
    with _API_INDEX_CACHE_LOCK:
        hit, cached = _cached_doc_answer(_API_INDEX_CACHE, _API_INDEX_REFUSALS, key, now)
    if hit:
        return cached
    try:
        raw = client.get_api_index()
    except Exception as e:
        logger.warning("/doc-bearer/index.json unavailable: %s: %s", type(e).__name__, e)
        return None
    index = build_api_index(raw) if isinstance(raw, dict) and "models" in raw else None
    with _API_INDEX_CACHE_LOCK:
        _store_doc_answer(_API_INDEX_CACHE, _API_INDEX_REFUSALS, key, now, index, _API_INDEX_CACHE_MAX_ENTRIES)
    return index


def clear_api_index_cache() -> None:
    """Drop every cached index and refusal (tests, credential rotation)."""
    with _API_INDEX_CACHE_LOCK:
        _API_INDEX_CACHE.clear()
        _API_INDEX_REFUSALS.clear()


def _categorize_error(error_msg: str) -> str:
    """Categorize an error message by its root cause."""
    error_lower = error_msg.lower()
    for category, info in ERROR_CATEGORIES.items():
        if category == "unknown":
            continue
        for pattern in info["patterns"]:
            if pattern in error_lower:
                return category
    return "unknown"


def _detect_domain_pattern(domain: List, model: Optional[str] = None) -> List[str]:
    """Detect patterns in domain that might cause issues."""
    patterns = []
    if not domain:
        return patterns

    domain_str = str(domain)

    # Detect dot notation (relational filters)
    if "." in domain_str and any(
        f".{field}" in domain_str
        for field in ["id", "name", "code", "state", "type", "partner", "company", "user", "product", "location"]
    ):
        patterns.append("dot_notation")

    # Detect complex OR conditions
    if domain_str.count("'|'") > 2 or domain_str.count('"|"') > 2:
        patterns.append("complex_or")

    # Detect negation
    if "'!'" in domain_str or '"!"' in domain_str:
        patterns.append("negation")

    # Detect 'any' operator (x2many search)
    if "'any'" in domain_str or '"any"' in domain_str:
        patterns.append("any_operator")

    # Detect child_of/parent_of (hierarchical)
    if "child_of" in domain_str or "parent_of" in domain_str:
        patterns.append("hierarchical")

    # Model-specific patterns
    if model == "stock.move.line":
        # picking_type_id with negative operators causes NotImplemented error
        if "picking_type_id" in domain_str:
            if any(op in domain_str for op in ["'!='", "'not in'", "'not like'", '"!="', '"not in"']):
                patterns.append("computed_field_negative_operator")
        # Deep related fields that cause issues
        if any(field in domain_str for field in ["product_category_name", "picking_code"]):
            patterns.append("deep_related_field")

    return patterns


def _detect_problematic_fields(fields: List, model: Optional[str] = None) -> List[str]:
    """Detect fields that might cause issues when included in search_read."""
    problematic = []
    if not fields:
        return problematic

    # Model-specific problematic fields (from source code analysis)
    model_problematic_fields = {
        "stock.move.line": {
            "non_stored_computed": ["lots_visible", "allowed_uom_ids"],
            "deep_related": ["product_category_name", "picking_code"],
            "computed_with_search": ["picking_type_id"],
        }
    }

    if model in model_problematic_fields:
        for category, field_list in model_problematic_fields[model].items():
            for field in field_list:
                if field in fields:
                    problematic.append(f"{field} ({category})")

    return problematic


def _track_model_issue(
    model: str, method: str, error_msg: str, domain: List = None, fields: List = None
) -> Dict[str, Any]:
    """
    Track a model/method issue with error categorization and pattern detection.
    Returns analysis with suggested solutions.
    """
    now = datetime.now().isoformat()
    category = _categorize_error(error_msg)
    domain_patterns = _detect_domain_pattern(domain, model) if domain else []
    problematic_fields = _detect_problematic_fields(fields, model) if fields else []

    with _RUNTIME_ISSUES_LOCK:
        if model not in RUNTIME_MODEL_ISSUES:
            RUNTIME_MODEL_ISSUES[model] = {}

        if method not in RUNTIME_MODEL_ISSUES[model]:
            RUNTIME_MODEL_ISSUES[model][method] = {"categories": {}, "first_seen": now, "total_count": 0}

        model_issues = RUNTIME_MODEL_ISSUES[model][method]
        model_issues["total_count"] += 1
        model_issues["last_seen"] = now

        # Track by category
        if category not in model_issues["categories"]:
            model_issues["categories"][category] = {
                "count": 0,
                "domain_patterns": {},
                "sample_errors": [],
                "solutions": ERROR_CATEGORIES[category]["solutions"],
            }

        cat_info = model_issues["categories"][category]
        cat_info["count"] += 1

        # Track domain patterns for this category
        for pattern in domain_patterns:
            cat_info["domain_patterns"][pattern] = cat_info["domain_patterns"].get(pattern, 0) + 1

        # Track problematic fields
        if "problematic_fields" not in cat_info:
            cat_info["problematic_fields"] = {}
        for field_info in problematic_fields:
            cat_info["problematic_fields"][field_info] = cat_info["problematic_fields"].get(field_info, 0) + 1

        # Keep last 3 unique error samples
        error_sample = error_msg[:300]
        if error_sample not in cat_info["sample_errors"]:
            cat_info["sample_errors"].append(error_sample)
            if len(cat_info["sample_errors"]) > 3:
                cat_info["sample_errors"].pop(0)

        total_count = model_issues["total_count"]
        occurrences = cat_info["count"]

    # Log detailed info (outside lock)
    logger.info(
        "Issue tracked: %s.%s | category=%s (%s) | domain_patterns=%s | "
        "problematic_fields=%s | total_occurrences=%d",
        model,
        method,
        category,
        ERROR_CATEGORIES[category]["cause"],
        domain_patterns or "none",
        problematic_fields or "none",
        total_count,
    )

    # Get model-specific recommendations from module_knowledge
    model_specific_advice = []
    if model in MODULE_KNOWLEDGE.get("model_limitations", {}):
        model_info = MODULE_KNOWLEDGE["model_limitations"][model]
        if method in model_info:
            method_info = model_info[method]
            if "safe_fields" in method_info:
                model_specific_advice.append(f"Safe fields: {', '.join(method_info['safe_fields'][:5])}...")
            if "avoid_in_domain" in method_info:
                model_specific_advice.append(f"Avoid in domain: {', '.join(method_info['avoid_in_domain'][:3])}")

    # Return analysis for immediate use
    return {
        "category": category,
        "cause": ERROR_CATEGORIES[category]["cause"],
        "domain_patterns": domain_patterns,
        "problematic_fields": problematic_fields,
        "solutions": ERROR_CATEGORIES[category]["solutions"],
        "model_specific_advice": model_specific_advice,
        "occurrences": occurrences,
    }


def get_error_suggestion(error_msg: str, model: Optional[str] = None, method: Optional[str] = None) -> Optional[str]:
    """Get a helpful suggestion based on error message patterns.

    Supports {model} template variable in suggestions (substituted with actual model name).
    """
    error_patterns = MODULE_KNOWLEDGE.get("error_patterns", {})
    model_name = model or "the model"

    def _substitute(suggestion: str) -> str:
        """Replace {model} placeholder with actual model name."""
        return suggestion.replace("{model}", model_name)

    error_lower = error_msg.lower()

    # Check for HTTP status code patterns
    for code, info in error_patterns.items():
        if code.startswith("_"):
            continue  # Skip meta keys like _fallback_patterns
        if code in str(error_msg):
            if isinstance(info, dict) and "patterns" in info:
                for pattern in info["patterns"]:
                    if pattern.get("match", "").lower() in error_lower:
                        return _substitute(pattern.get("suggestion", ""))
                # Return general suggestion for this code
                if "suggestion" in info:
                    return _substitute(info["suggestion"])
            elif isinstance(info, dict) and "suggestion" in info:
                return _substitute(info["suggestion"])

    # Check fallback patterns (match against any error regardless of HTTP code)
    fallback = error_patterns.get("_fallback_patterns", {})
    for pattern in fallback.get("patterns", []):
        if pattern.get("match", "").lower() in error_lower:
            return _substitute(pattern.get("suggestion", ""))

    # Check module-specific suggestions
    if model:
        for module_name, module_info in MODULE_KNOWLEDGE.get("modules", {}).items():
            if module_info.get("model") == model:
                # Check if using wrong method
                special_methods = module_info.get("special_methods", {})
                for special_method, method_info in special_methods.items():
                    if method_info.get("instead_of") == method:
                        example = method_info.get("example", "")
                        example_hint = f" Example: {example}" if example else ""
                        return f"Use {special_method}() instead of {method}() for {model}. {module_info.get('notes', '')}{example_hint}"

                # Check write_restrictions when write/create fails
                if method in ("write", "create") and module_info.get("write_restrictions"):
                    return module_info["write_restrictions"]

    return None


# ----- Standalone Helper Functions -----


def _get_module_knowledge() -> str:
    """Get the full module knowledge base with special methods and error patterns"""
    return _dumps(MODULE_KNOWLEDGE)


def _get_module_knowledge_by_name(module_name: str) -> str:
    """Get specific module knowledge"""
    modules = MODULE_KNOWLEDGE.get("modules", {})
    if module_name in modules:
        return _dumps({"module": module_name, **modules[module_name]})
    else:
        available = list(modules.keys())
        return _dumps({"error": f"Module '{module_name}' not found in knowledge base", "available_modules": available})


def _get_documentation_urls(target: str) -> str:
    """
    Get documentation URLs for a model or module.

    Returns:
    - Official docs URL
    - GitHub source URL
    - Suggested search queries
    - Special methods if known
    """
    ODOO_VERSION = "19.0"
    DOC_BASE = f"https://www.odoo.com/documentation/{ODOO_VERSION}"
    GITHUB_BASE = f"https://github.com/odoo/odoo/tree/{ODOO_VERSION}/addons"

    module_docs = MODULE_KNOWLEDGE.get("module_documentation", {})
    model_to_module = MODULE_KNOWLEDGE.get("model_to_module", {})
    modules_info = MODULE_KNOWLEDGE.get("modules", {})

    # Determine if target is a model or module
    is_model = "." in target and target not in module_docs

    if is_model:
        # It's a model - find its module
        module = model_to_module.get(target, target.split(".")[0])
        class_name = "".join(word.capitalize() for word in target.replace(".", "_").split("_"))
        filename = "_".join(target.split(".")) + ".py"
    else:
        module = target
        class_name = None
        filename = None

    result = {
        "target": target,
        "type": "model" if is_model else "module",
        "module": module,
        "documentation": {},
        "github": {},
        "search_queries": [],
        "special_methods": [],
    }

    # Get module documentation
    if module in module_docs:
        mod_info = module_docs[module]
        if mod_info.get("docs"):
            result["documentation"]["user_guide"] = f"{DOC_BASE}{mod_info['docs']}.html"
        if mod_info.get("github"):
            result["github"]["models"] = f"{GITHUB_BASE}{mod_info['github']}"
        if mod_info.get("snippets"):
            result["github"]["snippets"] = f"{GITHUB_BASE}{mod_info['snippets']}"

    # Common developer docs
    result["documentation"]["orm_reference"] = f"{DOC_BASE}/developer/reference/backend/orm.html"
    result["documentation"]["actions"] = f"{DOC_BASE}/developer/reference/backend/actions.html"

    # GitHub module root
    result["github"]["module_root"] = f"{GITHUB_BASE}/{module}"

    if filename:
        result["github"]["model_file_guess"] = f"{GITHUB_BASE}/{module}/models/{filename}"

    # Search queries
    result["search_queries"] = [
        f"site:github.com/odoo/odoo/blob/{ODOO_VERSION} {target}",
        f"site:odoo.com/documentation/{ODOO_VERSION} {module}",
    ]
    if class_name:
        result["search_queries"].append(f'site:github.com/odoo/odoo "class {class_name}"')

    # Include special methods if known
    for mod_name, mod_data in modules_info.items():
        if is_model and mod_data.get("model") == target:
            result["special_methods"] = list(mod_data.get("special_methods", {}).keys())
            if mod_data.get("notes"):
                result["notes"] = mod_data["notes"]
            break
        elif not is_model and mod_name == target:
            result["special_methods"] = list(mod_data.get("special_methods", {}).keys())
            if mod_data.get("notes"):
                result["notes"] = mod_data["notes"]
            break

    return _dumps(result)


# ----- Live fields_get cache (shared by the schema resources and the payload pre-flight) -----

# key: (client url, client username, model, attributes tuple | None) → (fetched_at, fields)
# The client identity is part of the key because fields_get is filtered by the
# caller's access rights — in multi-user mode two users must never share an entry.
_FieldsCacheKey = tuple[Any, Any, str, tuple[str, ...] | None]
_FIELDS_CACHE: "OrderedDict[_FieldsCacheKey, tuple[float, dict]]" = OrderedDict()
_FIELDS_CACHE_LOCK = threading.Lock()
_FIELDS_CACHE_TTL = 60  # seconds — shorter than _DOC_CACHE since model
# schemas can change with module updates
_FIELDS_CACHE_MAX = 100


def fields_cache_key(client: Any, model: str, attributes: Sequence[str] | None = None) -> _FieldsCacheKey:
    """Build the cache key for one (client, model, attribute subset) fetch."""
    return (*_client_identity(client), model, tuple(attributes) if attributes else None)


def fields_cache_get(key: _FieldsCacheKey) -> dict | None:
    """Return the cached fields for ``key`` if present and fresh, else ``None``."""
    now = time.time()
    with _FIELDS_CACHE_LOCK:
        cached = _FIELDS_CACHE.get(key)
        if cached and (now - cached[0]) < _FIELDS_CACHE_TTL:
            _FIELDS_CACHE.move_to_end(key)
            return cached[1]
    return None


def fields_cache_put(key: _FieldsCacheKey, fields: dict) -> None:
    """Store ``fields`` under ``key`` (LRU-bounded). Empty payloads are never cached."""
    if not fields or not isinstance(fields, dict):
        return
    with _FIELDS_CACHE_LOCK:
        _FIELDS_CACHE[key] = (time.time(), fields)
        _FIELDS_CACHE.move_to_end(key)
        while len(_FIELDS_CACHE) > _FIELDS_CACHE_MAX:
            _FIELDS_CACHE.popitem(last=False)


def get_fields_for_model(client: Any, model: str, attributes: Sequence[str] | None = None) -> dict:
    """Return the fields_get response for a model, with TTL+LRU caching.

    ``attributes`` narrows the request (and the cache key) to that subset —
    callers that only need type/required/readonly share one entry per model
    instead of each pulling the full definition.

    Empty responses are NOT cached — they typically indicate a silently-failed
    Odoo connection, and we don't want to grant a write token based on
    'no fields exist therefore validation passes'.
    """
    key = fields_cache_key(client, model, attributes)
    cached = fields_cache_get(key)
    if cached is not None:
        return cached

    # Cache miss or expired — fetch fresh.
    # Treat any exception (network failure, model-not-found, auth error) as
    # an empty schema — the caller's pre-flight will refuse to issue a
    # token without a verified field list, which is the safe behaviour.
    kwargs: dict[str, Any] = {"attributes": list(attributes)} if attributes else {}
    try:
        fields = client.execute_method(model, "fields_get", **kwargs)
    except Exception as exc:
        logger.warning("fields_get for %r failed: %s", model, exc)
        return {}

    if not fields or not isinstance(fields, dict):
        # Don't cache an empty (or unexpected non-dict) response — we don't
        # want to remember a failure.
        return {}

    fields_cache_put(key, fields)
    return cast(dict, fields)
