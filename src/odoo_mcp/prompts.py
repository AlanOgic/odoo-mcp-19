"""
MCP Prompt handlers for the Odoo MCP Server.

All 12 @mcp.prompt decorated functions. Importing this module
registers all prompts with the FastMCP instance.
"""

from fastmcp.prompts import Message

from .app import mcp

# ----- Discovery & Reference Prompts -----


@mcp.prompt(name="odoo-exploration")
def odoo_exploration_prompt() -> list[Message]:
    """Discover capabilities of this Odoo instance"""
    return [Message("""Explore this Odoo instance:

1. Read odoo://server/info for version and apps
2. Read odoo://workflows for available workflows
3. Read odoo://models for all models

Provide a summary of what's available.
""")]


@mcp.prompt(name="search-records")
def search_records_prompt(model: str = "res.partner") -> list[Message]:
    """Search for records in a model"""
    return [Message(f"""Search for records in {model}.

First read odoo://model/{model}/quick-schema to understand the fields.

Then use execute_method with:
- model='{model}'
- method='search_read'
- args_json='[[]]'  # empty domain for all records
- kwargs_json='{{"fields": ["name", "id"], "limit": 10}}'
""")]


@mcp.prompt(name="odoo-api-reference")
def api_reference_prompt() -> list[Message]:
    """Quick reference for Odoo API patterns"""
    return [Message("""## Odoo API Quick Reference

**Method Patterns:**
- search_read: `kwargs_json='{"domain": [...], "fields": [...], "limit": 100}'`
- create: `args_json='[{"field": "value"}]'`
- write: `args_json='[[ids], {"field": "value"}]'`
- unlink: `args_json='[[ids]]'`

**One2many/Many2many Commands:**
| Code | Meaning | Syntax |
|------|---------|--------|
| 0 | Create | `[0, 0, {values}]` |
| 1 | Update | `[1, id, {values}]` |
| 2 | Delete | `[2, id, 0]` |
| 3 | Unlink (M2M) | `[3, id, 0]` |
| 4 | Link (M2M) | `[4, id, 0]` |
| 5 | Unlink all (M2M) | `[5, 0, 0]` |
| 6 | Replace all (M2M) | `[6, 0, [ids]]` |

Commands are JSON lists, not Python tuples. Details: odoo://api/x2many-commands.

**Domain Operators:**
- Comparison: `=`, `!=`, `>`, `<`, `>=`, `<=`
- List: `in`, `not in`
- Text: `like`, `ilike`, `=like`, `=ilike`
- Logic: `&` (AND), `|` (OR), `!` (NOT)

**Domain Examples:**
```
[["state", "=", "draft"]]                    Simple
[["amount_total", ">", 1000]]                Comparison
[["name", "ilike", "test"]]                  Text search
["&", ["state", "=", "sale"], ["amount_total", ">", 500]]  AND
["|", ["state", "=", "draft"], ["state", "=", "sent"]]     OR
[["partner_id.name", "=", "Company"]]        Dot notation for related fields
```

**CRITICAL WARNINGS:**
- Many2one fields take the numeric ID, never the name string. To start from a name, let execute_method
  look it up: `resolve_json='{"partner_id": {"model": "res.partner", "search": "Acme"}}'` (errors on 0 or >1 matches).
- Read odoo://actions/{model} BEFORE calling unfamiliar models
- Check odoo://docs/{model} for documentation URLs

**Pre-execution Checklist:**
1. Model identified? (use odoo://find-model/{concept})
2. Actions verified? (read odoo://actions/{model})
3. Required fields known? (read odoo://model/{model}/quick-schema)
4. Types correct? (Many2one = ID, or a name through resolve_json)
""")]


@mcp.prompt(name="domain-builder")
def domain_builder_prompt(description: str = "") -> list[Message]:
    """Help construct complex domain filters"""
    return [Message(f"""## Domain Builder

Build a domain filter{' for: ' + description if description else ''}.

**Read odoo://domain-syntax for complete operator reference.**

**Key Operators:**
| Operator | Purpose | Example |
|----------|---------|---------|
| `=`, `!=` | Equality | `["state", "=", "draft"]` |
| `>`, `<`, `>=`, `<=` | Comparison | `["amount", ">", 1000]` |
| `in`, `not in` | List membership | `["state", "in", ["draft", "sent"]]` |
| `ilike` | Case-insensitive search | `["email", "ilike", "@gmail"]` |
| `child_of` | Hierarchical children | `["category_id", "child_of", 5]` |
| `parent_of` | Hierarchical parents | `["id", "parent_of", 10]` |
| `any` | x2many contains match | `["order_line", "any", [["product_id", "=", 1]]]` |

**Logic (Polish notation):**
- AND: `["&", term1, term2]`
- OR: `["|", term1, term2]`
- NOT: `["!", term]`

**Dot Notation for Related Fields:**
- `["partner_id.country_id.code", "=", "US"]`

**Example: Active US/CA companies with orders > $1000:**
```python
["&", "&", ["active", "=", true], ["is_company", "=", true],
 "|", ["country_id.code", "=", "US"], ["country_id.code", "=", "CA"]]
```

Use with execute_method search_read kwargs_json.
""")]


@mcp.prompt(name="hierarchical-query")
def hierarchical_query_prompt(model: str = "product.category") -> list[Message]:
    """Guide for querying parent/child tree structures"""
    return [Message(f"""## Hierarchical Query Guide for {model}

**Read odoo://hierarchical for complete patterns.**

**Query Patterns with execute_method:**

1. **Get all descendants (children + grandchildren):**
```
execute_method('{model}', 'search_read',
  kwargs_json='{{"domain": [["id", "child_of", PARENT_ID]], "fields": ["name", "parent_id"]}}')
```

2. **Get all ancestors (parents + grandparents):**
```
execute_method('{model}', 'search_read',
  kwargs_json='{{"domain": [["id", "parent_of", CHILD_ID]], "fields": ["name", "parent_id"]}}')
```

3. **Get direct children only:**
```
execute_method('{model}', 'search_read',
  kwargs_json='{{"domain": [["parent_id", "=", PARENT_ID]], "fields": ["name"]}}')
```

4. **Get root records (no parent):**
```
execute_method('{model}', 'search_read',
  kwargs_json='{{"domain": [["parent_id", "=", false]], "fields": ["name"]}}')
```
The children field has no fixed name (child_id on product.category, child_ids on res.partner):
take it, and the parent field, from odoo://model/{model}/quick-schema.

5. **Get tree path (if model has _parent_store):**
```
execute_method('{model}', 'search_read',
  kwargs_json='{{"domain": [["id", "=", ID]], "fields": ["name", "parent_path"]}}')
```
The parent_path field contains ancestor IDs separated by '/'.

**Common hierarchical models:**
- product.category (parent_id)
- account.account (parent_id)
- hr.department (parent_id)
- stock.location (location_id)
- knowledge.article (parent_id)
- documents.document (folder_id; folders have type = folder)
""")]


@mcp.prompt(name="paginated-search")
def paginated_search_prompt(model: str = "res.partner") -> list[Message]:
    """Guide for paginating large result sets"""
    return [Message(f"""## Paginated Search Guide for {model}

**Read odoo://pagination for complete reference.**

**Pattern: Get total count + paginated results**

Step 1: Get total count
```
execute_method('{model}', 'search_count',
  kwargs_json='{{"domain": YOUR_DOMAIN}}')
```

Step 2: Fetch page of results
```
execute_method('{model}', 'search_read',
  kwargs_json='{{
    "domain": YOUR_DOMAIN,
    "fields": ["name", "..."],
    "limit": 50,
    "offset": 0,
    "order": "name asc"
  }}')
```

**Pagination formula:**
- Page 1: offset=0, limit=50
- Page 2: offset=50, limit=50
- Page N: offset=(N-1)*limit, limit=50

**Iterate all records:**
1. Call search_read with `"limit": 100, "offset": 0` and a stable `"order": "id"`.
2. The records are in the `result` list of the response.
3. If `result` holds fewer than 100 records, that was the last page. Otherwise add 100 to `offset` and call again.

**Default limits:**
- MCP default: 100 records
- MCP maximum: 1000 records
- Use search_count first to know total
""")]


@mcp.prompt(name="aggregation-report")
def aggregation_report_prompt(model: str = "sale.order") -> list[Message]:
    """Guide for creating aggregation reports"""
    return [Message(f"""## Aggregation Report Guide for {model}

**Read odoo://aggregation for complete reference.**

**Use formatted_read_group (v19+). `read_group` is deprecated — do not use it.**
The param is `aggregates` (not `fields`); formatted_read_group always returns all
group levels (no `lazy` flag needed).

```
execute_method('{model}', 'formatted_read_group',
  kwargs_json='{{
    "domain": [DOMAIN],
    "groupby": ["field", ...],
    "aggregates": ["field:aggregator", ...]
  }}')
```

**Aggregators:**
| Aggregator | Purpose |
|------------|---------|
| `__count` | Count records |
| `sum` | Sum values |
| `avg` | Average |
| `min` | Minimum |
| `max` | Maximum |
| `count_distinct` | Distinct count |

**Date Grouping:**
- `:day`, `:week`, `:month`, `:quarter`, `:year`
- Example: `"groupby": ["create_date:month"]`

**Examples (written for sale.order; for another model take the field names from
odoo://model/{model}/quick-schema):**

1. Total by partner:
```
kwargs_json='{{"domain": [], "groupby": ["partner_id"], "aggregates": ["amount_total:sum"]}}'
```

2. Count by state:
```
kwargs_json='{{"domain": [], "groupby": ["state"], "aggregates": ["__count"]}}'
```

3. Monthly totals:
```
kwargs_json='{{"domain": [], "groupby": ["date_order:month"], "aggregates": ["amount_total:sum", "__count"]}}'
```

4. Multi-level grouping:
```
kwargs_json='{{"domain": [], "groupby": ["partner_id", "state"], "aggregates": ["amount_total:sum"]}}'
```
""")]


# ----- Business Workflow Prompts -----


@mcp.prompt(name="ar-aging-report")
def ar_aging_report_prompt() -> list[Message]:
    """Generate accounts receivable aging report"""
    return [Message("""Generate an AR aging report:

1. Open amount per customer, posted invoices only (see odoo://aggregation):
   execute_method("account.move", "formatted_read_group",
     kwargs_json='{"domain": [["move_type", "=", "out_invoice"], ["state", "=", "posted"],
                              ["payment_state", "in", ["not_paid", "partial"]]],
                   "groupby": ["partner_id"], "aggregates": ["amount_residual:sum"]}')

2. Due date of each open invoice, same domain (page with offset when 1000 records come back):
   execute_method("account.move", "search_read",
     kwargs_json='{"domain": [["move_type", "=", "out_invoice"], ["state", "=", "posted"],
                              ["payment_state", "in", ["not_paid", "partial"]]],
                   "fields": ["partner_id", "name", "invoice_date_due", "amount_residual"],
                   "order": "invoice_date_due", "limit": 1000}')

3. Compare invoice_date_due with today and put each amount_residual in a bucket:
   - Current (not yet due)
   - 1-30 days overdue
   - 31-60 days overdue
   - 61-90 days overdue
   - 90+ days overdue

4. Present as a formatted table with customer totals and recommendations.
""")]


@mcp.prompt(name="inventory-check")
def inventory_check_prompt(product: str = None) -> list[Message]:
    """Check inventory levels"""
    if product:
        return [Message(f"""Check inventory for "{product}":

1. Find the product: read odoo://find-model/product, then search product.product
2. Use execute_method with formatted_read_group (see odoo://aggregation):
   execute_method("stock.quant", "formatted_read_group",
     kwargs_json='{{"domain": [["product_id", "=", PRODUCT_ID], ["location_id.usage", "=", "internal"]],
                   "groupby": ["location_id"], "aggregates": ["quantity:sum", "reserved_quantity:sum"]}}')
   Keep the location_id.usage filter: without it customer, supplier, production and
   inventory-adjustment locations are summed with real stock.

3. Show on-hand and available (quantity minus reserved_quantity) by warehouse/location
""")]
    else:
        return [Message("""Check overall inventory status:

1. Use execute_method with formatted_read_group (see odoo://aggregation):
   execute_method("stock.quant", "formatted_read_group",
     kwargs_json='{"domain": [["location_id.usage", "=", "internal"]],
                   "groupby": ["product_id"], "aggregates": ["quantity:sum", "value:sum"]}')
   Keep the location_id.usage filter: without it customer, supplier, production and
   inventory-adjustment locations are summed with real stock.

2. Identify low stock items (quantity < reorder point)
3. Show top products by value
""")]


@mcp.prompt(name="crm-pipeline")
def crm_pipeline_prompt() -> list[Message]:
    """Analyze CRM pipeline"""
    return [Message("""Analyze the CRM pipeline:

1. Use execute_method with formatted_read_group (see odoo://aggregation):
   execute_method("crm.lead", "formatted_read_group",
     kwargs_json='{"domain": [["type", "=", "opportunity"]],
                   "groupby": ["stage_id"], "aggregates": ["expected_revenue:sum", "__count"]}')

2. Calculate conversion rates between stages
3. Identify opportunities that need attention:
   - Stuck in stage too long
   - High value opportunities
   - Upcoming activities

4. Present a pipeline summary with recommendations.
""")]


@mcp.prompt(name="customer-360")
def customer_360_prompt(customer: str) -> list[Message]:
    """Complete customer 360 view"""
    return [Message(f"""Get a 360-degree view of customer "{customer}":

1. Find the customer: read odoo://find-model/customer, then search res.partner

2. Get their data:
   - Basic info (name, email, phone, address)
   - Credit limit and receivables

3. Sales history (see odoo://aggregation):
   execute_method("sale.order", "formatted_read_group",
     kwargs_json='{{"domain": [["partner_id", "=", CUSTOMER_ID]],
                   "groupby": ["state"], "aggregates": ["amount_total:sum"]}}')

4. Invoice status:
   execute_method("account.move", "formatted_read_group",
     kwargs_json='{{"domain": [["partner_id", "=", CUSTOMER_ID], ["move_type", "=", "out_invoice"]],
                   "groupby": ["payment_state"], "aggregates": ["amount_residual:sum"]}}')

5. Recent activities:
   - Messages and notes from mail.message

6. CRM opportunities:
   - Open opportunities from crm.lead

Present a comprehensive customer profile with key insights.
""")]


@mcp.prompt(name="daily-operations")
def daily_operations_prompt() -> list[Message]:
    """Daily operations dashboard"""
    return [Message("""Generate a daily operations summary:

**Sales:**
- New orders today (use formatted_read_group on sale.order, see odoo://aggregation)
- Pending quotations needing follow-up

**Inventory:**
- Pending deliveries (stock.picking with state = assigned or waiting)
- Low stock alerts

**Accounting:**
- Invoices to send (draft invoices)
- Overdue payments (use AR aging prompt)
- Cash position

**CRM:**
- Activities due today
- Hot opportunities (high probability, high value)

Use execute_method with formatted_read_group for efficient aggregation. Present as a dashboard.
""")]
