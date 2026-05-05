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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from .constants import (
    _DOC_CACHE,
    _DOC_CACHE_LOCK,
    _DOC_CACHE_MAX_ENTRIES,
    _DOC_CACHE_TTL,
    ERROR_CATEGORIES,
    MODULE_KNOWLEDGE,
    RUNTIME_MODEL_ISSUES,
    _RUNTIME_ISSUES_LOCK,
)
from .odoo_client import get_odoo_client

# ----- Compact Schema Builder -----


def _build_compact_schema(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Build an ultra-compact schema representation from fields_get output.

    Returns a dict with:
    - fields: {name: {t, req?, ro?, rel?, sel?}} with short keys
    - required_fields: list of required field names (top-level for quick access)

    This produces ~1.5-2KB vs 5-10KB for /fields, saving 60-80% tokens.
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


def _get_live_doc(model_name: str) -> Optional[Dict[str, Any]]:
    """Fetch live model docs from /doc-bearer/ with in-memory caching.

    Returns the doc dict if available, or None on any failure.
    Failures are silent -- the caller falls back to static data.
    """
    now = time.time()
    with _DOC_CACHE_LOCK:
        if model_name in _DOC_CACHE:
            ts, data = _DOC_CACHE[model_name]
            if now - ts < _DOC_CACHE_TTL:
                return data

    try:
        odoo = get_odoo_client()
        doc = odoo.get_model_doc(model_name)
        if doc and isinstance(doc, dict) and "methods" in doc:
            with _DOC_CACHE_LOCK:
                _DOC_CACHE[model_name] = (now, doc)
                if len(_DOC_CACHE) > _DOC_CACHE_MAX_ENTRIES:
                    oldest = min(_DOC_CACHE, key=lambda k: _DOC_CACHE[k][0])
                    del _DOC_CACHE[oldest]
            return doc
    except Exception as e:
        logger.warning("/doc-bearer/ unavailable for %s: %s", model_name, e)

    return None


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


def _detect_domain_pattern(domain: List, model: str = None) -> List[str]:
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


def _detect_problematic_fields(fields: List, model: str = None) -> List[str]:
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
        "occurrences": cat_info["count"],
    }


def get_error_suggestion(error_msg: str, model: str = None, method: str = None) -> Optional[str]:
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
    return json.dumps(MODULE_KNOWLEDGE, indent=2)


def _get_module_knowledge_by_name(module_name: str) -> str:
    """Get specific module knowledge"""
    modules = MODULE_KNOWLEDGE.get("modules", {})
    if module_name in modules:
        return json.dumps({"module": module_name, **modules[module_name]}, indent=2)
    else:
        available = list(modules.keys())
        return json.dumps(
            {"error": f"Module '{module_name}' not found in knowledge base", "available_modules": available}, indent=2
        )


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

    return json.dumps(result, indent=2)


# ----- Live fields_get cache (used by payload pre-flight) -----

_FIELDS_CACHE: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_FIELDS_CACHE_LOCK = threading.Lock()
_FIELDS_CACHE_TTL = 60  # seconds — shorter than _DOC_CACHE since model
# schemas can change with module updates
_FIELDS_CACHE_MAX = 100


def get_fields_for_model(client, model: str) -> dict:
    """Return the fields_get response for a model, with TTL+LRU caching.

    Empty responses are NOT cached — they typically indicate a silently-failed
    Odoo connection, and we don't want to grant a write token based on
    'no fields exist therefore validation passes'.
    """
    now = time.time()
    with _FIELDS_CACHE_LOCK:
        cached = _FIELDS_CACHE.get(model)
        if cached and (now - cached[0]) < _FIELDS_CACHE_TTL:
            _FIELDS_CACHE.move_to_end(model)
            return cached[1]

    # Cache miss or expired — fetch fresh.
    # Treat any exception (network failure, model-not-found, auth error) as
    # an empty schema — the caller's pre-flight will refuse to issue a
    # token without a verified field list, which is the safe behaviour.
    try:
        fields = client.execute_method(model, "fields_get")
    except Exception as exc:
        logger.warning("fields_get for %r failed: %s", model, exc)
        return {}

    if not fields or not isinstance(fields, dict):
        # Don't cache an empty (or unexpected non-dict) response — we don't
        # want to remember a failure.
        return {}

    with _FIELDS_CACHE_LOCK:
        _FIELDS_CACHE[model] = (now, fields)
        _FIELDS_CACHE.move_to_end(model)
        while len(_FIELDS_CACHE) > _FIELDS_CACHE_MAX:
            _FIELDS_CACHE.popitem(last=False)
    return fields
