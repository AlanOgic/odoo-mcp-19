---
name: cyanview-inventory-watchdog
description: >
  Monitor Cyanview stock levels, flag low-inventory products, identify slow-moving
  items, predict stockouts from open sales orders, and suggest replenishment actions.
  Queries stock.quant, product.product, stock.warehouse.orderpoint, sale.order,
  and purchase.order to build a complete inventory health picture.
  Triggers on: "stock check", "inventory", "stock levels", "what do we have in stock",
  "low stock", "reorder", "replenish", "stockout", "are we running low",
  "what's in the warehouse", "stock dashboard", "inventory report",
  "do we have enough [product]", "how many [product] left", "stock alert",
  "purchase order needed", "what should we reorder", "slow movers",
  "dead stock", "inventory health", "stock value", "warehouse status",
  or any question about product availability, stock quantities, or replenishment
  — even casual questions like "do we have RIOs" or "can we ship 5 CI0s today".
allowed-tools: mcp__odoo19-mcp__execute_method, mcp__odoo19-mcp__batch_execute, mcp__odoo19-mcp__read_resource
---

# Cyanview Inventory Watchdog

Monitor stock health across all Cyanview warehouses. This skill answers the question
every ops person asks daily: "do we have enough, and what do we need to order?"

## Why this matters

Cyanview ships hardware worldwide — RIOs, RCPs, CI0s, cables, accessories. A stockout
means a delayed customer order and potentially a lost deal. Overstocking ties up cash.
This skill gives instant visibility into what's available, what's committed, what's
incoming, and what needs attention.

## Cyanview stock locations

| ID | Name | Purpose |
|----|------|---------|
| 8 | CY/Stock | Main warehouse (Belgium) — primary shipping location |
| 32 | CY/Bulk | Bulk storage |
| 77 | CY/Spare | Spare parts for repairs |
| 44 | CY/Loan | Demo kit loan inventory |
| 37 | CY/Return | Returns staging |
| 48 | CY/Repair | Devices currently in repair |
| 81 | UK/Stock | UK warehouse |

For stock availability calculations, the relevant locations are **CY/Stock (8)** and
**UK/Stock (81)** — these are where orders ship from. CY/Loan and CY/Repair hold
committed devices that aren't available for sale.

## Workflow

### Step 1: Understand the request

The user might ask anything from a broad dashboard to a specific product check.
Classify the request:

| Request type | What to fetch |
|-------------|---------------|
| "stock check" / "inventory dashboard" | Full overview: all CY- products with quantities |
| "do we have [product]" / "how many X left" | Single product availability across locations |
| "low stock" / "what needs reordering" | Products below reorder point or with alerts |
| "stock value" | Quantities × unit cost for valuation |
| "slow movers" / "dead stock" | Products with no outgoing moves in 90+ days |
| "can we ship X units of Y" | Check available_quantity vs requested amount |
| "what's incoming" | Pending purchase orders and incoming transfers |
| "stockout risk" | Cross-reference on-hand vs open SO demand |

### Step 2: Fetch product stock data

The most efficient approach is to query `product.product` directly — it has computed
fields that aggregate stock across locations:

```
model: product.product
method: search_read
kwargs_json: {
  "domain": [["default_code", "ilike", "CY-"], ["is_storable", "=", true]],
  "fields": ["id", "name", "default_code", "qty_available", "free_qty",
             "virtual_available", "incoming_qty", "outgoing_qty",
             "reordering_min_qty", "reordering_max_qty",
             "standard_price", "list_price", "monthly_demand",
             "categ_id", "tracking"],
  "order": "default_code asc",
  "limit": 100
}
```

**Key fields explained:**
- `qty_available` — total on-hand across all internal locations
- `free_qty` — on-hand minus reserved (what's actually available to promise)
- `virtual_available` — forecasted: on-hand + incoming - outgoing
- `incoming_qty` — from confirmed purchase orders / incoming transfers
- `outgoing_qty` — committed to confirmed sales orders / outgoing transfers
- `reordering_min_qty` / `reordering_max_qty` — from reorder rules (0 if none set)
- `monthly_demand` — Odoo's computed average monthly demand
- `standard_price` — unit cost (for valuation)

### Step 3: Fetch per-location breakdown (when needed)

For detailed warehouse view, query `stock.quant`:

```
model: stock.quant
method: search_read
kwargs_json: {
  "domain": [
    ["product_id.default_code", "ilike", "CY-"],
    ["location_id", "in", [8, 32, 77, 44, 48, 81]],
    ["quantity", ">", 0]
  ],
  "fields": ["product_id", "location_id", "quantity", "reserved_quantity",
             "available_quantity"],
  "order": "product_id asc",
  "limit": 200
}
```

This shows exactly where each unit sits — useful for answering "how many RIOs are
in Belgium vs UK" or "how many devices are out on loan right now."

### Step 4: Check reorder rules

```
model: stock.warehouse.orderpoint
method: search_read
kwargs_json: {
  "domain": [["active", "=", true]],
  "fields": ["product_id", "product_min_qty", "product_max_qty",
             "qty_on_hand", "qty_forecast", "qty_to_order",
             "location_id", "trigger", "supplier_id"],
  "order": "product_id asc",
  "limit": 50
}
```

Products with `qty_on_hand < product_min_qty` need immediate reordering.
Products with `qty_to_order > 0` have Odoo-computed replenishment needs.

### Step 5: Check open demand (sales orders)

To predict stockouts, fetch confirmed but unshipped sales orders:

```
model: sale.order.line
method: search_read
kwargs_json: {
  "domain": [
    ["order_id.state", "=", "sale"],
    ["product_id.default_code", "ilike", "CY-"],
    ["qty_to_deliver", ">", 0]
  ],
  "fields": ["product_id", "product_uom_qty", "qty_delivered",
             "qty_to_deliver", "order_id", "order_partner_id"],
  "order": "product_id asc",
  "limit": 100
}
```

Compare `qty_to_deliver` per product against `free_qty`. If total pending delivery
exceeds free stock, flag a stockout risk.

### Step 6: Check incoming supply (purchase orders)

```
model: purchase.order.line
method: search_read
kwargs_json: {
  "domain": [
    ["order_id.state", "in", ["purchase", "done"]],
    ["product_id.default_code", "ilike", "CY-"],
    ["qty_received", "<", "product_qty"]
  ],
  "fields": ["product_id", "product_qty", "qty_received",
             "order_id", "date_planned"],
  "order": "date_planned asc",
  "limit": 50
}
```

This shows what's on order and when it's expected. Cross-reference with demand to
determine if incoming supply will cover the gap.

### Step 7: Identify slow movers (optional, on request)

Query `stock.move` for products with no outgoing moves in the last 90 days:

```
model: stock.move
method: formatted_read_group
kwargs_json: {
  "domain": [
    ["product_id.default_code", "ilike", "CY-"],
    ["state", "=", "done"],
    ["date", ">=", "90_DAYS_AGO"],
    ["location_dest_id.usage", "=", "customer"]
  ],
  "groupby": ["product_id"],
  "aggregates": ["__count"]
}
```

Note: `read_group` is deprecated in Odoo 19 — use `formatted_read_group`, whose
measure param is `aggregates` (not `fields`); `["__count"]` returns the per-group count.

Compare the list of products that had outgoing moves against the full product catalog.
Products with on-hand stock but no recent outgoing moves are slow movers or dead stock.

## Output formats

### Full inventory dashboard

```
## Inventory Dashboard — {date}

### Stock summary
| Product | Code | On Hand | Free | Reserved | Incoming | Outgoing | Forecast | Status |
|---------|------|---------|------|----------|----------|----------|----------|--------|
| RIO | CY-RIO | 45 | 32 | 13 | 10 | 8 | 47 | ✅ OK |
| RCP Joystick | CY-RCP-J | 3 | 1 | 2 | 0 | 5 | -2 | 🔴 STOCKOUT RISK |
| Camera Interface | CY-CI0 | 18 | 15 | 3 | 0 | 2 | 16 | ✅ OK |

### Alerts
🔴 **CY-RCP-J**: Only 1 free unit, 5 pending delivery. Forecast goes negative.
   → Suggest PO for 10 units (max qty from reorder rule)

⚠️ **CY-VP4**: Below reorder point (on hand: 2, min: 5)
   → Reorder rule suggests ordering 8 units

### Stock by location
| Location | CY-RIO | CY-RCP-J | CY-CI0 | CY-VP4 |
|----------|--------|----------|--------|--------|
| CY/Stock (BE) | 35 | 2 | 15 | 1 |
| UK/Stock | 8 | 1 | 3 | 1 |
| CY/Loan | 2 | 0 | 0 | 0 |

### Stock value
Total inventory value: €{sum of qty × standard_price}
Top 3 by value: {products with highest total_value}
```

### Single product check

```
## CY-RCP-J — Stock Status

On hand: 3 | Free: 1 | Reserved: 2
Incoming: 0 (no open PO)
Outgoing: 5 (from 3 confirmed SOs)
Forecast: -2 🔴

**Where it is:**
- CY/Stock: 2 (1 reserved)
- UK/Stock: 1 (1 reserved)

**Pending deliveries:**
- SO2603-4535 → NBCUniversal: 2 units
- SO2603-4524 → Fletcher Group: 2 units
- SO2602-4411 → Riedel: 1 unit

**Reorder rule:** Min 5 / Max 15 → Needs 14 units
**Suggested action:** Create PO for 14× CY-RCP-J immediately
```

### Stockout risk report

```
## Stockout Risk Report — {date}

Products where open demand exceeds free stock:

| Product | Free | Pending Delivery | Gap | Incoming | Net Risk |
|---------|------|-----------------|-----|----------|----------|
| CY-RCP-J | 1 | 5 | -4 | 0 | 🔴 -4 |
| CY-LIC-RIO-WAN | 12 | 58 | -46 | 50 | ⚠️ -8 (after PO) |

**Immediate action needed:**
1. CY-RCP-J: No incoming supply. Create PO now.
2. CY-LIC-RIO-WAN: PO covers most but still short 8 units.
```

## Status indicators

Use these consistently in all outputs:

- ✅ **OK** — free stock covers forecast demand
- ⚠️ **Low** — below reorder point but positive forecast
- 🔴 **Stockout risk** — forecast goes negative or free < pending delivery
- 💤 **Slow mover** — on-hand stock but no outgoing moves in 90+ days
- 📦 **Incoming** — below min but PO exists to cover

## Adapt to context

| User says | Response style |
|-----------|---------------|
| "quick stock check" | Summary table, alerts only |
| "full inventory report" | Complete dashboard with location breakdown and value |
| "do we have 5 RIOs" | Direct yes/no with availability details |
| "what should we order" | Reorder recommendations with suggested quantities |
| "stock value" | Quantities × cost with totals |

## Follow-up actions

After presenting the inventory view, suggest relevant next steps:

- Low stock → "Want me to draft a purchase order?" (connects to odoo-expert skill)
- Stockout risk → "Should I check which sales orders are affected?"
- Slow movers → "Want me to check when these last sold?"
- Loan devices → "Should I check the loan lifecycle status?" (connects to loan-lifecycle skill)
- Devices in repair → "Want the RMA dashboard?" (connects to RMA skill)

## Edge cases

- **Licences vs hardware**: CY-LIC-* products are digital licences. They may show
  as "storable" but don't occupy physical space. Note this distinction in reports
  and skip them from location breakdowns unless specifically asked.

- **Kit products**: Some products are BOM kits (e.g., a "starter pack"). Their stock
  depends on component availability. If `is_kits = true`, note it and check component
  stock instead.

- **Negative quantities**: `stock.quant` can show negative quantities from
  backorders or inventory discrepancies. Flag these as data quality issues:
  "⚠️ Negative stock detected for {product} at {location} — needs inventory adjustment."

- **No reorder rule**: If a storable product has no orderpoint (reordering_min_qty = 0
  and reordering_max_qty = 0), note it: "No reorder rule set — consider adding one."

- **Multi-warehouse**: Always show Belgium (CY/Stock) and UK separately. Customers in
  the UK/EU ship from different locations, so combined totals can be misleading.

- **Rental/loan devices**: Devices at CY/Loan (44) are committed to active loans.
  They show in `qty_available` but NOT in `free_qty`. Make this clear when users
  ask about total vs available stock.
