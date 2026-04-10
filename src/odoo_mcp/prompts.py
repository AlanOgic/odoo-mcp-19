"""
MCP Prompt handlers for the Odoo MCP Server.

All 13 @mcp.prompt decorated functions. Importing this module
registers all prompts with the FastMCP instance.
"""

from .app import mcp
from fastmcp.prompts import Message


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

First read odoo://model/{model}/schema to understand the fields.

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
| 0 | Create | `(0, 0, {values})` |
| 1 | Update | `(1, id, {values})` |
| 2 | Delete | `(2, id, 0)` |
| 3 | Unlink (M2M) | `(3, id, 0)` |
| 4 | Link (M2M) | `(4, id, 0)` |
| 5 | Unlink all (M2M) | `(5, 0, 0)` |
| 6 | Replace all (M2M) | `(6, 0, [ids])` |

**Domain Operators:**
- Comparison: `=`, `!=`, `>`, `<`, `>=`, `<=`
- List: `in`, `not in`
- Text: `like`, `ilike`, `=like`, `=ilike`
- Logic: `&` (AND), `|` (OR), `!` (NOT)

**Domain Examples:**
```python
[("state", "=", "draft")]                    # Simple
[("amount", ">", 1000)]                      # Comparison
[("name", "ilike", "%test%")]                # Text search
["&", ("state", "=", "sale"), ("amount", ">", 500)]  # AND
["|", ("state", "=", "draft"), ("state", "=", "sent")]  # OR
[("partner_id.name", "=", "Company")]        # Dot notation for related fields
```

**CRITICAL WARNINGS:**
- Many2one fields = ALWAYS use numeric ID, never the name string!
- Read odoo://actions/{model} BEFORE calling unfamiliar models
- Check odoo://docs/{model} for documentation URLs

**Pre-execution Checklist:**
1. Model identified? (use odoo://find-model/{concept})
2. Actions verified? (read odoo://actions/{model})
3. Required fields known? (read odoo://model/{model}/schema)
4. Types correct? (Many2one = ID, not name)
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
  kwargs_json='{{"domain": [["parent_id", "=", false]], "fields": ["name", "child_id"]}}')
```

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
```python
offset = 0
limit = 100
while True:
    results = execute_method('{model}', 'search_read',
      kwargs_json=f'{{"domain": [], "limit": {{limit}}, "offset": {{offset}}}}')
    if len(results) < limit:
        break  # Last page
    offset += limit
```

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

**Using read_group with execute_method:**

```
execute_method('{model}', 'read_group',
  args_json='[DOMAIN]',
  kwargs_json='{{
    "fields": ["field:aggregator", ...],
    "groupby": ["field", ...]
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

**Examples for {model}:**

1. Total by partner:
```
kwargs_json='{{"fields": ["amount_total:sum"], "groupby": ["partner_id"]}}'
```

2. Count by state:
```
kwargs_json='{{"fields": ["__count"], "groupby": ["state"]}}'
```

3. Monthly totals:
```
kwargs_json='{{"fields": ["amount_total:sum", "__count"], "groupby": ["date_order:month"]}}'
```

4. Multi-level grouping:
```
kwargs_json='{{"fields": ["amount_total:sum"], "groupby": ["partner_id", "state"]}}'
```

**Note:** read_group is deprecated in v19. Current MCP uses it for compatibility.
New code should use formatted_read_group (web module).
""")]


# ----- Business Workflow Prompts -----


@mcp.prompt(name="quote-to-cash")
def quote_to_cash_prompt(order_id: str = None) -> list[Message]:
    """Complete quote-to-cash workflow"""
    if order_id:
        return [Message(f"""Execute the quote-to-cash workflow for order {order_id}:

Use execute_workflow("quote_to_cash", '{{"order_id": {order_id}}}')

This will:
1. Confirm the sales order (action_confirm)
2. Create invoice (_create_invoices)
3. Post invoice (action_post)

Report the result of each step.
""")]
    else:
        return [Message("""Guide me through creating a complete sales flow:

1. First, find or create a customer (res.partner)
2. Create a quotation (sale.order) with order lines
3. Confirm the quotation
4. Create and post the invoice
5. Optionally register payment

Read odoo://tools/sales for available operations.
""")]


@mcp.prompt(name="ar-aging-report")
def ar_aging_report_prompt() -> list[Message]:
    """Generate accounts receivable aging report"""
    return [Message("""Generate an AR aging report:

1. Use execute_method with read_group (see odoo://aggregation):
   execute_method("account.move", "read_group",
     args_json='[[["move_type", "=", "out_invoice"], ["payment_state", "in", ["not_paid", "partial"]]]]',
     kwargs_json='{"fields": ["amount_residual:sum"], "groupby": ["partner_id"]}')

2. Then categorize by aging buckets:
   - Current (not yet due)
   - 1-30 days overdue
   - 31-60 days overdue
   - 61-90 days overdue
   - 90+ days overdue

3. Present as a formatted table with customer totals and recommendations.
""")]


@mcp.prompt(name="inventory-check")
def inventory_check_prompt(product: str = None) -> list[Message]:
    """Check inventory levels"""
    if product:
        return [Message(f"""Check inventory for "{product}":

1. Find the product: read odoo://find-model/product, then search product.product
2. Use execute_method with read_group (see odoo://aggregation):
   execute_method("stock.quant", "read_group",
     args_json='[[["product_id", "=", PRODUCT_ID]]]',
     kwargs_json='{{"fields": ["quantity:sum"], "groupby": ["location_id"]}}')

3. Show available quantity by warehouse/location
""")]
    else:
        return [Message("""Check overall inventory status:

1. Use execute_method with read_group (see odoo://aggregation):
   execute_method("stock.quant", "read_group", args_json='[[]]',
     kwargs_json='{"fields": ["quantity:sum", "value:sum"], "groupby": ["product_id"]}')

2. Identify low stock items (quantity < reorder point)
3. Show top products by value
""")]


@mcp.prompt(name="crm-pipeline")
def crm_pipeline_prompt() -> list[Message]:
    """Analyze CRM pipeline"""
    return [Message("""Analyze the CRM pipeline:

1. Use execute_method with read_group (see odoo://aggregation):
   execute_method("crm.lead", "read_group",
     args_json='[[["type", "=", "opportunity"]]]',
     kwargs_json='{"fields": ["expected_revenue:sum", "__count"], "groupby": ["stage_id"]}')

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
   execute_method("sale.order", "read_group",
     args_json='[[["partner_id", "=", CUSTOMER_ID]]]',
     kwargs_json='{{"fields": ["amount_total:sum"], "groupby": ["state"]}}')

4. Invoice status:
   execute_method("account.move", "read_group",
     args_json='[[["partner_id", "=", CUSTOMER_ID], ["move_type", "=", "out_invoice"]]]',
     kwargs_json='{{"fields": ["amount_residual:sum"], "groupby": ["payment_state"]}}')

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
- New orders today (use read_group on sale.order, see odoo://aggregation)
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

Use execute_method with read_group for efficient aggregation. Present as a dashboard.
""")]
