---
name: cyanview-serial-tracker
description: >
  Trace any Cyanview device by serial number across its full lifecycle in Odoo:
  owner, sale order, deliveries, RMA/repair history, current stock location, and
  chatter log. The "device passport" — one query to know everything about a unit.
  Triggers on: "serial number", "SN", "track device", "device history",
  "where is this unit", "who has SN", "device passport", "trace", "numéro de série",
  "find device", "lookup serial", "what happened to", or any mention of a Cyanview
  serial number (format CY-XXX-NN-NNN). Also triggers when a serial is mentioned in
  the context of support, RMA, or customer questions — even casual like "who bought
  CY-RIO-15-42" or "has this CI0 been repaired before".
allowed-tools: mcp__odoo19-mcp__execute_method, mcp__odoo19-mcp__batch_execute, mcp__odoo19-mcp__read_resource
argument-hint: "[serial-number]"
---

# Cyanview Serial Tracker

Trace any Cyanview device through its complete lifecycle using its serial number (lot number
in Odoo). This is the "device passport" — everything you need to know about a specific unit,
from when it was created to where it is right now, in one consolidated view.

## Why this matters

When a customer calls about a device, or support needs to check warranty status, or sales
wants to know if a unit was previously loaned — the answers are scattered across stock.lot,
sale.order, repair.order, stock.picking, and chatter messages. Hunting through Odoo tabs
wastes time and risks missing critical context (like "this unit was already repaired twice
for the same issue"). This skill pulls everything together in seconds.

## Serial number format

Cyanview serial numbers follow the pattern: `CY-{PRODUCT}-{BATCH}-{NUMBER}`

Examples:
- `CY-RIO-15-42` — RIO, batch 15, unit 42
- `CY-CI0-12-10` — CI0, batch 12, unit 10
- `CY-RCP-40-174` — RCP, batch 40, unit 174

The user might provide the full serial, a partial match, or just mention a number in
conversation. Be flexible — search with `ilike` when the input is partial.

## Workflow

### Step 1: Find the serial number (stock.lot)

Parse the user's message for anything that looks like a serial number. Then search:

```
model: stock.lot
method: search_read
kwargs_json: {
  "domain": [["name", "ilike", "<serial>"]],
  "fields": ["id", "name", "product_id", "ref", "partner_ids",
             "sale_order_ids", "sale_order_count",
             "purchase_order_ids", "purchase_order_count",
             "repair_line_ids", "in_repair_count", "repaired_count",
             "delivery_ids", "delivery_count",
             "product_qty", "location_id",
             "create_date", "note"],
  "limit": 10
}
```

If multiple lots match (partial search), list them and ask the user to pick one.
If no match, try broader search (remove batch prefix, search just the number portion).

### Step 2: Gather linked records (parallel queries)

Once you have the lot ID, fire these queries in parallel using `batch_execute`:

**A. Sale orders** — who bought it, when, for how much:
```
model: sale.order
method: search_read
args_json: null
kwargs_json: {
  "domain": [["id", "in", <sale_order_ids>]],
  "fields": ["name", "partner_id", "date_order", "state", "amount_total",
             "currency_id", "invoice_status", "user_id"]
}
```

**B. Purchase orders** — where we sourced it (if applicable):
```
model: purchase.order
method: search_read
kwargs_json: {
  "domain": [["id", "in", <purchase_order_ids>]],
  "fields": ["name", "partner_id", "date_order", "state", "amount_total"]
}
```

**C. Deliveries** — shipping history:
```
model: stock.picking
method: search_read
kwargs_json: {
  "domain": [["id", "in", <delivery_ids>]],
  "fields": ["name", "partner_id", "picking_type_id", "state",
             "scheduled_date", "date_done", "origin", "carrier_id",
             "carrier_tracking_ref"]
}
```

**D. Repair orders** — RMA/repair history (search by lot_id):
```
model: repair.order
method: search_read
kwargs_json: {
  "domain": [["lot_id", "=", <lot_id>]],
  "fields": ["name", "partner_id", "state", "schedule_date", "under_warranty",
             "product_id", "x_studio_cyanview_status", "x_studio_fault_description",
             "x_studio_customer_reference", "internal_notes", "create_date",
             "sale_order_id", "ticket_id"]
}
```

**E. Current stock location** — where is the device right now:
```
model: stock.quant
method: search_read
kwargs_json: {
  "domain": [["lot_id", "=", <lot_id>]],
  "fields": ["location_id", "quantity", "reserved_quantity"]
}
```

**F. Partners linked to this lot** — customers who have/had it:
```
model: res.partner
method: search_read
kwargs_json: {
  "domain": [["id", "in", <partner_ids>]],
  "fields": ["name", "email", "phone", "country_id", "is_company", "parent_id"]
}
```

**G. Chatter / recent messages** — last 10 messages on the lot:
```
model: mail.message
method: search_read
kwargs_json: {
  "domain": [["model", "=", "stock.lot"], ["res_id", "=", <lot_id>]],
  "fields": ["date", "author_id", "body", "subtype_id", "message_type"],
  "order": "date desc",
  "limit": 10
}
```

### Step 3: Check for active loans

If any linked sale order might be a loan/rental, check for associated project tasks:

```
model: sale.order
method: search_read
kwargs_json: {
  "domain": [["id", "in", <sale_order_ids>]],
  "fields": ["name", "project_ids"]
}
```

If `project_ids` is not empty, fetch the project tasks to see loan status:
```
model: project.task
method: search_read
kwargs_json: {
  "domain": [["project_id", "in", <project_ids>]],
  "fields": ["name", "stage_id", "date_deadline", "partner_id", "kanban_state"],
  "order": "date_deadline asc"
}
```

### Step 4: Determine warranty status

A device is typically under warranty if:
- It was sold less than 2 years ago (standard Cyanview warranty)
- OR an extended warranty was purchased (CY-WAR-EXT-2Y = +2 years, CY-WAR-EXT-3Y = +3 years)

Check the sale order lines for warranty products:
```
model: sale.order.line
method: search_read
kwargs_json: {
  "domain": [["order_id", "in", <sale_order_ids>],
             ["product_id.default_code", "in", ["CY-WAR-EXT-2Y", "CY-WAR-EXT-3Y"]]],
  "fields": ["product_id", "order_id"]
}
```

Calculate warranty expiry:
- Standard: sale date + 2 years
- With CY-WAR-EXT-2Y: sale date + 4 years
- With CY-WAR-EXT-3Y: sale date + 5 years

### Step 5: Present the device passport

Format the output as a clear, structured summary. Adapt the level of detail to context —
if the user just asked "who has CY-RIO-15-42", give them the owner quickly and offer to
dig deeper. If they asked for the full history, go all out.

**Full device passport format:**

```
## Device Passport: CY-RIO-15-42

**Product**: CY-RIO (Remote Camera Interface)
**Serial**: CY-RIO-15-42
**Created**: 2024-03-15
**Current location**: [location name]
**Current owner**: [partner name]

### Warranty
- Sale date: 2024-06-01
- Warranty type: Standard (2 years) / Extended (+2Y / +3Y)
- Warranty expires: 2026-06-01
- Status: Active / Expired

### Sales History
| Order | Customer | Date | Amount | Status |
|-------|----------|------|--------|--------|
| SO123 | Company X | 2024-06-01 | €1,200 | Delivered |

### Delivery History
| Transfer | Type | Date | Tracking | Status |
|----------|------|------|----------|--------|
| WH/OUT/456 | Delivery | 2024-06-05 | DHL 123456 | Done |

### Repair History (RMA)
| RMA | Customer | Date | Fault | Status | Warranty |
|-----|----------|------|-------|--------|----------|
| RMA2501-015 | Company X | 2025-01-10 | Ethernet port stuck | Repaired | Yes |

### Loan History
[If applicable — project tasks linked to rental SOs]

### Recent Activity
[Last 5-10 chatter messages, summarized]

### Quick Links
- Lot: https://<domain>/odoo/inventory/products/lots/<lot_id>
- [SO link, repair link, etc.]
```

### Step 6: Offer follow-up actions

Based on what the passport reveals, suggest relevant next steps:
- "This device has been repaired twice for similar issues — want me to create an RMA?"
  → hand off to cyanview-rma skill
- "The warranty expires next month — want me to notify the customer about extension?"
- "This unit is currently in a demo loan — want to check the loan status?"
  → hand off to cyanview-loan-lifecycle skill
- "Want me to pull up the full customer briefing?"
  → hand off to cyanview-customer-360 skill

## Handling edge cases

- **Serial not found**: suggest checking for typos, try partial match, or search by product type
- **Multiple matches**: list them with product type and owner, let user pick
- **Device never sold** (still in stock): show location and available quantity
- **Device scrapped**: show scrap location and last known history
- **Legacy products** (RIO-LIVE, RCP-DUO, etc.): still trackable, just note it's a legacy product
- **No repair history**: simply omit that section — don't show empty tables
- **Lot with no linked data**: sometimes lots are created during manufacturing but not yet shipped. Show what's known and note it's "in production/stock"

## Odoo URL format

For generating clickable links:
- Lot: `https://cylan2.odoo.com/odoo/inventory/products/lots/{lot_id}`
- Sale order: `https://cylan2.odoo.com/odoo/sales/{so_id}`
- Repair: `https://cylan2.odoo.com/odoo/repairs/{repair_id}`
- Delivery: `https://cylan2.odoo.com/odoo/inventory/delivery-orders/{picking_id}`
- Partner: `https://cylan2.odoo.com/odoo/contacts/{partner_id}`
