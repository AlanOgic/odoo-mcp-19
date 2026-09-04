---
name: cyanview-customer-360
description: >
  Build a complete 360° briefing for any Cyanview customer or contact using Odoo MCP.
  Aggregates partner info, sales orders, invoices, CRM opportunities, RMA/repairs,
  demo kit loans, support history, and recent chatter into a single actionable overview.
  Triggers on: "tell me about [customer]", "customer 360", "customer overview",
  "client summary", "briefing for [company]", "what do we know about [contact]",
  "prep for call with [name]", "customer history", "account review",
  "fiche client", "résumé client", or any request to pull together everything
  we know about a customer, contact, or company — even if they just mention
  a company name in a context that implies wanting background info.
allowed-tools: mcp__odoo19-mcp__execute_method, mcp__odoo19-mcp__batch_execute, mcp__odoo19-mcp__read_resource
argument-hint: "[company-name]"
---

# Cyanview Customer 360°

Pull together everything we know about a customer into one concise, actionable briefing.
This is the "prep for a call in 2 minutes" skill — it queries multiple Odoo models in
parallel and surfaces what matters: open deals, pending invoices, active repairs, loan
status, and recent communication.

## Why this matters

Before any customer call, email, or meeting, the team needs context. Hunting through
Odoo tabs one by one wastes time and risks missing something. A 360° view catches
things like: "this customer has an overdue invoice AND an open RMA AND a stale
opportunity — tread carefully on the upsell."

## Workflow

### Step 1: Find the customer

The user will mention a company name, contact name, email, or Odoo partner ID.
Search broadly to find the right record:

```
model: res.partner
method: search_read
kwargs_json: {
  "domain": ["|", "|", "|",
    ["name", "ilike", "SEARCH_TERM"],
    ["email", "ilike", "SEARCH_TERM"],
    ["ref", "ilike", "SEARCH_TERM"],
    ["vat", "ilike", "SEARCH_TERM"]
  ],
  "fields": ["id", "name", "email", "phone", "mobile", "street", "city",
             "country_id", "parent_id", "child_ids", "is_company",
             "commercial_partner_id", "ref", "vat", "website",
             "user_id", "category_id", "comment", "credit_limit",
             "total_invoiced", "credit"],
  "limit": 10
}
```

**Disambiguation**: If multiple results, present a short list and ask the user to pick.
If one result is clearly the right one (exact name match, or it's a company and the
others are its contacts), proceed with that.

**Company vs contact**: If the result is a contact (has `parent_id`), pull the company
too. If it's a company, also note key contacts (`child_ids`). The briefing covers the
**commercial entity** — use `commercial_partner_id` as the anchor for all cross-model
queries, so you catch orders placed by any contact under that company.

### Step 2: Gather data (use batch_execute for speed)

Once you have the partner ID (and commercial_partner_id for company-level queries),
fetch data from all relevant models. Use `batch_execute` to run these in parallel
when possible — it's faster and uses fewer round-trips.

Here's what to fetch. Not every customer will have data in every model — that's fine,
just skip empty sections in the output.

#### 2a. Sales orders

```
model: sale.order
method: search_read
kwargs_json: {
  "domain": [["partner_id.commercial_partner_id", "=", COMMERCIAL_PARTNER_ID]],
  "fields": ["id", "name", "state", "date_order", "amount_total",
             "invoice_status", "partner_id", "user_id",
             "is_rental_order", "rental_status"],
  "order": "date_order desc",
  "limit": 20
}
```

Classify orders:
- **Open/confirmed**: state = "sale"
- **Quotations**: state in ("draft", "sent")
- **Rental/loans**: is_rental_order = true
- **Completed**: state = "done"

#### 2b. Invoices & payments

```
model: account.move
method: search_read
kwargs_json: {
  "domain": [
    ["partner_id.commercial_partner_id", "=", COMMERCIAL_PARTNER_ID],
    ["move_type", "=", "out_invoice"],
    ["state", "=", "posted"]
  ],
  "fields": ["id", "name", "date", "amount_total", "amount_residual",
             "payment_state", "invoice_date_due"],
  "order": "date desc",
  "limit": 15
}
```

Flag:
- **Overdue**: `payment_state != "paid"` AND `invoice_date_due < today`
- **Partially paid**: `payment_state = "partial"`
- **Total invoiced**: sum of all `amount_total`
- **Outstanding balance**: sum of `amount_residual` where `payment_state != "paid"`

#### 2c. CRM opportunities

```
model: crm.lead
method: search_read
kwargs_json: {
  "domain": [
    ["partner_id.commercial_partner_id", "=", COMMERCIAL_PARTNER_ID],
    ["active", "=", true]
  ],
  "fields": ["id", "name", "stage_id", "expected_revenue", "probability",
             "user_id", "date_deadline", "tag_ids", "activity_ids",
             "create_date", "date_last_stage_update"],
  "order": "create_date desc",
  "limit": 10
}
```

Flag:
- **Stale**: `date_last_stage_update` > 30 days ago with stage not Won/Lost
- **High value**: `expected_revenue` > 10,000 EUR
- **Won deals**: stage is Won (shows purchase intent history)
- **Lost deals**: check lost_reason for context

#### 2d. Repairs / RMAs

```
model: repair.order
method: search_read
kwargs_json: {
  "domain": ["|",
    ["partner_id", "=", PARTNER_ID],
    ["partner_id.parent_id", "=", COMMERCIAL_PARTNER_ID]
  ],
  "fields": ["id", "name", "product_id", "lot_id", "state",
             "x_studio_cyanview_status", "x_studio_fault_description",
             "under_warranty", "create_date", "tag_ids"],
  "order": "create_date desc",
  "limit": 10
}
```

Flag:
- **Active RMAs**: state not in ("done", "cancel")
- **Repeat repairs**: same serial number appearing multiple times
- **Warranty vs paid**: note the split

#### 2e. Demo kit loans

```
model: sale.order
method: search_read
kwargs_json: {
  "domain": [
    ["partner_id.commercial_partner_id", "=", COMMERCIAL_PARTNER_ID],
    ["is_rental_order", "=", true]
  ],
  "fields": ["id", "name", "state", "rental_status", "date_order",
             "x_studio_planned_return", "next_action_date", "amount_total"],
  "order": "date_order desc",
  "limit": 5
}
```

Flag:
- **Active loan**: rental_status = "return"
- **Overdue return**: x_studio_planned_return < today AND rental_status = "return"
- **Returned**: rental_status = "returned"

#### 2f. Recent communication (chatter)

```
model: mail.message
method: search_read
kwargs_json: {
  "domain": [
    ["res_id", "=", PARTNER_ID],
    ["model", "=", "res.partner"],
    ["message_type", "in", ["email", "comment"]]
  ],
  "fields": ["id", "date", "subject", "body", "author_id", "message_type"],
  "order": "date desc",
  "limit": 5
}
```

Also check recent messages on their sales orders and leads for broader context.

#### 2g. Scheduled activities

```
model: mail.activity
method: search_read
kwargs_json: {
  "domain": [
    ["res_model", "in", ["res.partner", "sale.order", "crm.lead"]],
    ["partner_id", "=", PARTNER_ID]
  ],
  "fields": ["id", "activity_type_id", "summary", "date_deadline",
             "user_id", "res_model", "res_id", "note"],
  "order": "date_deadline asc",
  "limit": 10
}
```

#### 2h. Products purchased (from SO lines)

```
model: sale.order.line
method: search_read
kwargs_json: {
  "domain": [
    ["order_id.partner_id.commercial_partner_id", "=", COMMERCIAL_PARTNER_ID],
    ["order_id.state", "in", ["sale", "done"]]
  ],
  "fields": ["product_id", "product_uom_qty", "price_subtotal", "order_id"],
  "limit": 50
}
```

Aggregate to show: which CY- products they own, quantities, and total spend per product.
This is gold for upsell identification — e.g., they have 5 RIOs but no RCP-J.

### Step 3: Compose the briefing

Present everything in a structured, scannable format. The goal is: someone glances at
this for 60 seconds before a call and knows the full picture.

Use this template:

```
## 360° — {Company Name}

**Contact**: {name} — {email} — {phone}
**Address**: {city}, {country}
**Salesperson**: {user_id name}
**Customer ref**: {ref} | **VAT**: {vat}
{if comment: **Internal note**: {comment}}

---

### Revenue snapshot
Total invoiced: €{total_invoiced}
Outstanding: €{outstanding} {if overdue: ⚠ €{overdue_amount} overdue}
Last order: {date} ({SO name}, €{amount})

### Products owned
{table: Product | Qty | Total}
CY-RIO        | 5  | €5,000
CY-RCP-J      | 1  | €2,500
CY-LIC-WAN-5  | 1  | €500
{upsell hint if applicable: "💡 5 RIOs but only LAN licence — WAN upgrade opportunity?"}

### Open business
{for each active SO/quotation:}
- **{SO name}** ({state}) — €{amount} — {date}
{for each active CRM lead:}
- **{lead name}** — Stage: {stage} — €{expected_revenue} — {days in stage}d
  {if stale: ⚠ No movement in {days} days}

### Repairs & RMA
{if any:}
| RMA | Device | Serial | Status | Fault | Warranty |
|-----|--------|--------|--------|-------|----------|
{rows}
{if no repairs: "No repair history — clean record."}

### Demo kit loans
{if any:}
| SO | Shipped | Return due | Status |
|----|---------|------------|--------|
{rows}
{if active loan overdue: ⚠ Return overdue by {days} days}

### Recent communication
{last 3-5 messages with date, author, and brief content}
{if no recent comms: "⚠ No communication in the last 30 days"}

### Pending activities
{list of scheduled activities with dates and assignees}

### Flags & action items
{Synthesized list of things that need attention:}
- ⚠ Overdue invoice: {invoice} — €{amount} — {days} days past due
- ⚠ Stale opportunity: {lead} — no movement in {days} days
- ⚠ Active RMA: {rma} — {status}
- 💡 Upsell: {product suggestion based on owned products}
- ⚠ Demo kit overdue: {loan SO} — {days} days past return date
```

### Step 4: Adapt to context

The briefing template above is the full version. Adapt based on what the user needs:

| User says | Focus on |
|-----------|----------|
| "prep for call with X" | Full briefing with emphasis on flags & action items |
| "what has X purchased" | Products owned + sales history, skip RMA/loans |
| "any issues with X" | RMA + overdue invoices + stale leads |
| "revenue from X" | Invoice totals + SO history + pipeline value |
| "history with X" | Chronological: first order → latest, include won/lost deals |

When the request is clearly focused on one area, still include a brief mention of
other areas if there's something flagged (e.g., "Note: there's also an overdue RMA
for this customer").

### Step 5: Offer follow-up actions

After presenting the briefing, suggest relevant next steps based on what you found:

- Overdue invoice → "Want me to draft a payment reminder?"
- Stale opportunity → "Should I update the CRM lead or schedule a follow-up?"
- Active RMA → "Need the repair status details? I can pull the full RMA history."
- Upsell opportunity → "Want me to create a quote for the upgrade?"
- Active loan → "Want me to check the loan lifecycle tasks?"
- No recent communication → "Should I draft a check-in email?"

These connect naturally to the other Cyanview skills (cyanview-quote, cyanview-rma,
cyanview-lead-qualifier, cyanview-loan-lifecycle).

## Efficiency notes

**Minimizing API calls**: The queries in Step 2 can be parallelized using `batch_execute`.
Group the read-only queries into a single batch call. This typically cuts the total
query time from 6-8 sequential calls to 1-2 batch calls.

**Token management**: Some customers have extensive history. Use `limit` on all queries
and focus on the most recent / most relevant records. For products purchased, aggregate
rather than listing every SO line.

**commercial_partner_id**: This is the key to getting complete data. A company may have
many contacts who place orders independently. Using `commercial_partner_id` ensures you
catch everything under the company umbrella, not just the specific contact.

## Edge cases

- **Contact without company**: Use the contact's own ID as both partner and commercial
  partner. The briefing covers their individual history.

- **Huge customer** (100+ orders): Summarize rather than list. Show totals, last 5
  orders, and focus on what's currently active. Mention "showing last 5 of {total} orders."

- **Brand new customer** (no history): Say so clearly: "New contact — no order history,
  no RMA, no prior communication. First touch." Check if there's a CRM lead that brought
  them in.

- **Duplicate partners**: If search returns what looks like duplicates (same company,
  different records), note it: "⚠ Possible duplicate records for this company — IDs
  {x} and {y}. Consider merging in Odoo."

- **Multiple commercial entities**: Some groups have subsidiaries that are separate
  commercial partners. If the user asks about "NBC" and you find both "NBCUniversal
  Media, LLC" and "NBC Sports", clarify which one.

- **No email on partner**: Flag it — "⚠ No email address on file for this contact."
  This affects ability to send quotes, invoices, etc.
