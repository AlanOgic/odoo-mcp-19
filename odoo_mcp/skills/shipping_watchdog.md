---
name: cyanview-shipping-watchdog
description: >
  Audit Cyanview confirmed sale orders that haven't been fully shipped yet. Identifies stuck
  outbound shipments, overdue rental returns, licence-only orders without pickings, and
  searches chatter for delay/backorder explanations. Produces a prioritized action list.
  Use this skill whenever the user mentions: "shipping status", "unshipped orders",
  "delivery check", "what hasn't shipped", "stuck shipments", "overdue deliveries",
  "shipment audit", "delivery health", "pending shipments", "check deliveries",
  "what's waiting to ship", "shipping delays", "backorders", "picking status",
  "confirmed but not shipped", "overdue returns", "rental returns", or any request
  to review the state of outbound logistics and delivery follow-ups — even casual
  like "anything stuck in shipping?" or "what do we still need to send out?"
---

# Cyanview Shipping Watchdog

## Purpose

Find every confirmed sale order that isn't fully delivered, classify each one by type
(real shipment, rental return, licence/service, stuck backorder), check for existing
delay explanations in chatter, and present a clear action list sorted by urgency.

This skill covers the full "delivery health" picture — from outbound shipments to
rental equipment awaiting customer return.

## Workflow

### Step 1: Fetch confirmed SOs not fully shipped

Start with recent orders (current year and previous year) to keep the results actionable.
If the user asks about older orders or "all time", expand the date range.

```
model: sale.order
method: search_read
kwargs_json: {
  "domain": [
    ["state", "=", "sale"],
    ["delivery_status", "!=", "full"],
    ["date_order", ">=", "YYYY-01-01"]
  ],
  "fields": ["id", "name", "state", "partner_id", "date_order", "amount_total",
             "delivery_status", "picking_ids", "currency_id", "note",
             "commitment_date", "is_rental_order", "rental_status"],
  "order": "date_order desc",
  "limit": 80
}
```

Replace `YYYY` with the previous year (e.g., if today is 2026, use 2025-01-01).

### Step 2: Separate orders by category

From the results, classify each SO into one of these buckets:

| Category | How to detect | What it means |
|----------|--------------|---------------|
| **Has active pickings** | `picking_ids` is non-empty | Physical delivery in progress or stuck |
| **Rental return** | `is_rental_order == true` and `rental_status == "return"` | Equipment shipped, waiting for customer to return it |
| **No pickings at all** | `picking_ids` is empty, `delivery_status` is false | Likely a licence/service order, or delivery not yet created |
| **Partial delivery** | `delivery_status == "partial"` | Some items shipped, remainder outstanding |

This classification is critical — without it, rental returns look like stuck outbound
shipments and licence orders look like forgotten deliveries.

### Step 3: Fetch picking details for orders with pickings

For SOs that have `picking_ids`, fetch the non-done pickings to see what's actually stuck:

```
model: stock.picking
method: search_read
kwargs_json: {
  "domain": [
    ["sale_id", "in", [LIST_OF_SO_IDS_WITH_PICKINGS]],
    ["state", "!=", "done"]
  ],
  "fields": ["id", "name", "sale_id", "state", "scheduled_date", "note",
             "origin", "partner_id", "backorder_id", "repair_ids"],
  "order": "scheduled_date asc",
  "limit": 80
}
```

From the picking data, extract:

| Field | What it tells you |
|-------|-------------------|
| `name` | Picking reference — **CY/IN/** prefix = incoming (rental return or repair) |
| `state` | `assigned` = ready to process, `confirmed` = waiting on stock, `waiting` = blocked by another operation |
| `scheduled_date` | When it was supposed to happen — compare to today for overdue detection |
| `backorder_id` | If set, this picking is a remainder from a partial shipment |
| `origin` | May reference another operation (e.g., "Return of WH/RET/xxxxx") |
| `repair_ids` | If non-empty, this picking is linked to a repair order |
| `note` | Rarely filled but check anyway |

**Picking name prefixes** (Cyanview-specific):
- `SH` or `WH/OUT` = outbound shipment
- `CY/IN` = incoming receipt — almost always a rental/loan return
- `WH/RET` = customer return

### Step 4: Confirm rental returns via SO

For any picking with a `CY/IN/` prefix, verify by checking the parent SO:

```
model: sale.order
method: search_read
kwargs_json: {
  "domain": [["id", "in", [LIST_OF_SO_IDS_WITH_CY_IN_PICKINGS]]],
  "fields": ["id", "name", "is_rental_order", "rental_status", "partner_id"]
}
```

If `is_rental_order == true` and `rental_status == "return"`, confirm it's a rental
return — the equipment was already delivered, and the picking represents the expected
return from the customer. These are NOT stuck outbound shipments.

### Step 5: Search chatter for delay explanations

For each SO that has overdue or stuck pickings (excluding confirmed rental returns),
search the chatter for context. Check both the SO and its pickings:

**On the sale order:**
```
model: mail.message
method: search_read
kwargs_json: {
  "domain": [
    ["res_id", "in", [LIST_OF_OVERDUE_SO_IDS]],
    ["model", "=", "sale.order"],
    ["message_type", "in", ["email", "comment"]]
  ],
  "fields": ["res_id", "body", "date", "author_id", "subject"],
  "order": "res_id desc, date desc",
  "limit": 100
}
```

If the result is too large (Odoo returns an error for oversized responses), split into
smaller batches of 3-5 SO IDs at a time.

**On the pickings:**
```
model: mail.message
method: search_read
kwargs_json: {
  "domain": [
    ["res_id", "in", [LIST_OF_PICKING_IDS]],
    ["model", "=", "stock.picking"],
    ["message_type", "in", ["email", "comment"]]
  ],
  "fields": ["res_id", "body", "date", "author_id"],
  "order": "res_id desc, date desc",
  "limit": 50
}
```

**What to look for in chatter:**
- Keywords: delay, backorder, stock, wait, hold, out of stock, ordered, ETA, consolidate
- Internal notes from team (Laurent, Thierry, Renaud, Alan, Mike, Nicolas)
- Customer requests to hold/delay/consolidate shipments
- References to other orders being combined
- Mentions of items being on order from suppliers

**Parsing HTML bodies:** Messages come as HTML. Strip tags mentally or use a regex
to extract text content. Focus on substance, ignore signatures and formatting.

### Step 6: Classify each overdue situation

For each SO/picking that's overdue, assign one of these verdicts:

| Verdict | Criteria | Icon |
|---------|----------|------|
| **Explained — customer hold** | Customer asked to hold/consolidate | ✅ |
| **Explained — out of stock** | Team noted item on order, customer informed | ✅ |
| **Explained — planned delay** | Team noted a future delivery date | ✅ |
| **Explained — rental return** | `is_rental_order` + `rental_status == "return"` | 🔄 |
| **Partially explained** | Internal notes exist but customer NOT informed | ⚠️ |
| **Unexplained** | No chatter, no notes, no context | ❌ |
| **Data cleanup needed** | Cancelled pickings, orphaned operations, wrong stage | 🧹 |
| **Licence/service — no shipping needed** | No pickings, low amount, likely digital delivery | 📋 |

### Step 7: Present results

#### Summary table (always show this first)

Group by category and sort by urgency (unexplained first, then partially explained,
then explained, then rental returns):

```
## Shipping Health Check — [date]

### 🔴 Stuck Outbound Shipments (unexplained)
| SO | Customer | Amount | Picking | Scheduled | Overdue | Notes |
|----|----------|--------|---------|-----------|---------|-------|
| SO2602-4513 | OneDiversified | $1,901 | SH2603-03698 | Mar 2 | 32 days | Backorder, customer asked overnight |

### ⚠️ Partially Explained (needs follow-up)
| SO | Customer | Amount | Picking | Scheduled | Context |
|----|----------|--------|---------|-----------|---------|
| SO2603-4578 | OneDiversified | $522 | SH2603-03744 | Mar 18 | Laurent: "2-3 weeks" on Mar 17 — now past |

### ✅ Explained Delays
| SO | Customer | Amount | Reason |
|----|----------|--------|--------|
| SO2602-4411 | Riedel | 106,152€ | 2nd batch planned for July |

### 🔄 Overdue Rental Returns
| SO | Customer | Amount | Return Due | Overdue |
|----|----------|--------|-----------|---------|
| SO2512-4246 | Bayside Church | $163 | Jan 15 | 78 days |

### 📋 Licence/Service Orders (no physical shipping)
[Count] orders with no pickings — likely licence upgrades or digital deliveries.
Notable high-value ones: [list any over 2,000€]
```

#### Detail drill-down

After the summary table, for any item the user wants to investigate further, provide
the full chatter timeline (like the Teltec 126-BE-02698 example from our session).

### Step 8: Suggest actions

For each category, recommend concrete next steps:

| Category | Suggested action |
|----------|-----------------|
| Unexplained stuck | "Investigate immediately — check stock, contact warehouse, inform customer" |
| Partially explained (past ETA) | "Follow up on restock status, update customer" |
| Explained but old | "Verify the explanation is still valid — things may have changed" |
| Overdue rental return | "Contact customer for return ETA, check if equipment is still needed" |
| Data cleanup | "Close/cancel the orphaned picking, update SO stage" |
| Licence orders | "Verify licence was activated — if yes, mark as delivered" |

## Edge Cases

- **Very large result sets:** If more than 80 SOs match, focus on the last 6 months
  first and offer to go further back if needed.
- **Chatter too large:** Odoo may return an error for messages exceeding token limits.
  Split into batches of 3-5 SO IDs, or search with keyword filters (body ilike "delay").
- **sale.order.line access:** The `search_read` method on `sale.order.line` sometimes
  fails with a 500 error. Use `read` with explicit line IDs from `order_line` field instead.
- **Picking with no SO:** Some pickings may be standalone (manual stock moves). Skip
  these unless the user specifically asks about them.
- **Multiple pickings per SO:** An SO can have many pickings (outbound + return + backorder).
  Show only the non-done ones unless the user asks for the full history.
- **Currency:** Display amounts with the currency from the SO (`currency_id`). Don't
  convert — just show EUR or USD as stored.
- **Placeholder dates (Dec 31):** Scheduled dates set to Dec 31 of the current year
  are usually placeholder dates for long-term loans. Flag them as "long-term" rather
  than overdue.
- **Old confirmed SOs (pre-2024):** These are often historical data issues — orders
  that were fulfilled but not properly closed in Odoo. Mention them in a footnote
  but don't alarm the user about them.
