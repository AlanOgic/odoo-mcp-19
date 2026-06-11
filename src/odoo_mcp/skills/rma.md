---
name: cyanview-rma
description: >
  Manage Cyanview RMA/repair orders in Odoo. Triggers on: "RMA", "repair", "return",
  "warranty", "DOA", "broken", "faulty", "defective", "reparation", "retour",
  "fix device", "serial number issue", "device not working", "create RMA",
  "repair status", "what's in repair", or any request involving device returns and repairs.
---

# Cyanview RMA / Repair Manager

## Purpose

Create, track, and manage repair orders for Cyanview devices. Handles the full
lifecycle: customer reports a fault → **Clorag diagnostic** → RMA created → device
arrives → diagnostic → repair → return → close.

## Clorag integration (RAG-assisted diagnostics)

Before creating an RMA, **always** query Clorag first (clorag-mcp server, ~30 native tools).
This step often resolves the issue without shipping hardware.

### Diagnostic sequence

**Step 1 — Search similar resolved cases:**
```
Tool: search_support_cases
Query: "RIO ethernet port not responding"
```
→ Returns matching support threads with problem/solution summaries.

**Step 2 — Get solution details (if match found):**
```
Tool: get_support_case
case_id: [from step 1 results]
```
→ Returns full case: summary, problem description, resolution, product involved.

**Step 3 — Check device/camera info if relevant:**
```
Tool: get_camera  (if "camera not detected" type fault)
Tool: search      (general RAG query for broader context)
```

### Decision flow

1. Query `search_support_cases` with fault description
2. If similar case found with remote fix → share troubleshooting steps with customer, **NO RMA**
3. If similar case shows hardware failure → proceed to create RMA
4. If no match, use `search` for general RAG context
5. If fault is "camera not detected" → use `get_camera` / `search_cameras` first — may be
   wrong cable or unsupported camera, not a faulty device

### Example: "RIO ethernet port dead"

```
search_support_cases("RIO ethernet not working")
→ Case #247: "Dirty ethernet connector. Quick clean up fixed it."
→ Case #189: "Firmware bug in 25.8.x. Update to 25.9.1 resolved."

Decision: Suggest cleaning + firmware update before RMA.
Only create RMA if customer confirms issue persists after troubleshooting.
```

## Data model

### repair.order — Key fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | char (auto) | RMA number, auto-generated: `RMA{YYMM}-{SEQ}` |
| `product_id` | many2one → product.product | The device being repaired |
| `lot_id` | many2one → stock.lot | Serial number of the device |
| `partner_id` | many2one → res.partner | Customer (contact, not company) |
| `state` | selection | Odoo state: draft / confirmed / under_repair / done / cancel |
| `x_studio_cyanview_status` | selection | **Cyanview workflow** (see below) |
| `x_studio_fault_description` | char | Customer-reported fault (short text) |
| `x_studio_customer_reference` | char | Customer's own reference / ticket number |
| `x_studio_support_source` | char | Link to support thread (Google Groups) |
| `x_studio_internal_notes` | html | Additional internal notes |
| `internal_notes` | html | Diagnostic + Repair notes (structured) |
| `under_warranty` | boolean | Warranty flag |
| `tag_ids` | many2many → repair.tags | Repair tags |
| `location_id` | many2one → stock.location | Always **CY/Repair** (id: 48) |
| `user_id` | many2one → res.users | Technician assigned |
| `ticket_id` | many2one → helpdesk.ticket | Linked helpdesk ticket |
| `schedule_date` | datetime | Scheduled repair date |
| `priority` | selection | 0 (normal) or 1 (urgent) |
| `picking_id` | many2one → stock.picking | Incoming shipment |

### Cyanview status workflow (x_studio_cyanview_status)

```
1-new (Accepted) → 2-arrived → 3-confirmed → 4-underRepair → 5-repaired → 6-returned → 7-done
                                                                                          ↓
                                                                                     8-cancelled
```

| Value | Label | When |
|-------|-------|------|
| `1-new` | Accepted | RMA created, waiting for device to arrive |
| `2-arrived` | Arrived | Device received at CY/Repair |
| `3-confirmed` | Confirmed | Fault confirmed after initial inspection |
| `4-underRepair` | Under Repair | Technician working on it |
| `5-repaired` | Repaired | Repair complete, ready to return |
| `6-returned` | Returned | Device shipped back to customer |
| `7-done` | Done | Fully closed (customer confirmed receipt) |
| `8-cancelled` | Cancelled | RMA cancelled |

### Repair tags (repair.tags)

| ID | Name | When to use |
|----|------|-------------|
| 1 | Warranty | Device under warranty |
| 2 | DOA | Dead on arrival (new device, immediate failure) |
| 3 | Free Replacement | Replacement sent at no charge |
| 6 | 50% Replacement | Half-price replacement offered |
| 8 | Spare parts | Parts used from spare stock |
| 10 | Paid repair | Customer pays for repair |
| 11 | Verification | Device sent for verification (no confirmed fault) |
| 12 | Shipped Back | Device returned to customer |

### Stock locations

| ID | Name | Usage |
|----|------|-------|
| 48 | CY/Repair | All repairs happen here |
| 8 | CY/Stock | Main stock (Belgium) |
| 32 | CY/Bulk | Bulk storage |
| 77 | CY/Spare | Spare parts |
| 44 | CY/Loan | Loan devices |
| 37 | CY/Return | Returns staging |
| 81 | UK/Stock | UK warehouse |

### Team

| ID | Name | Role |
|----|------|------|
| 8 | Paul Rathgeb | Main repair technician |
| 23 | Nicolas Fasseaux | Backup technician |
| 7 | David Bourgeois | Sales (handles some RMAs) |

## Workflows

### 1. Create a new RMA

**Trigger**: "create RMA for [customer] — [device] [serial] — [fault description]"

#### Step 1: Identify the device

Search by serial number first (most reliable):
```
model: stock.lot
method: search_read
kwargs_json: {"domain": [["name", "ilike", "SERIAL"]], "fields": ["id", "name", "product_id"], "limit": 5}
```

If serial unknown, search by product:
```
model: product.product
method: search_read
kwargs_json: {"domain": [["default_code", "ilike", "CY-DEV-"]], "fields": ["id", "name", "default_code", "tracking"], "limit": 10}
```

Common devices in repair:
- CY-RIO (id: 10) — most common
- CY-RCP-J (id: 274) — second most common
- CY-RCP-DUO (id: 78)
- CY-RCP-DUO-J (id: 275)
- CY-CI0 (id: 8)
- CY-VP4 (id: 11)

#### Step 2: Identify the customer

Search partner (same logic as cyanview-quote skill):
```
model: res.partner
method: search_read
kwargs_json: {"domain": ["|", ["name", "ilike", "SEARCH"], ["email", "ilike", "SEARCH"]], "fields": ["id", "name", "email", "parent_id", "country_id"], "limit": 5}
```

#### Step 3: Determine warranty

- Check `under_warranty` based on context (user tells you, or check purchase date)
- If warranty: tag "Warranty" (id: 1)
- If DOA (new device, failed immediately): tag "DOA" (id: 2) + "Warranty" (id: 1)
- If out of warranty: tag "Paid repair" (id: 10) unless user says otherwise

#### Step 4: Create the repair order

```json
{
  "model": "repair.order",
  "method": "create",
  "args_json": "[{
    \"product_id\": PRODUCT_ID,
    \"lot_id\": LOT_ID,
    \"partner_id\": PARTNER_ID,
    \"location_id\": 48,
    \"x_studio_cyanview_status\": \"1-new\",
    \"x_studio_fault_description\": \"SHORT FAULT DESCRIPTION\",
    \"x_studio_support_source\": \"SUPPORT_THREAD_URL\",
    \"x_studio_customer_reference\": \"CUSTOMER_REF\",
    \"under_warranty\": true/false,
    \"tag_ids\": [[4, TAG_ID]],
    \"user_id\": 8,
    \"schedule_date\": \"YYYY-MM-DD HH:MM:SS\",
    \"priority\": \"0\"
  }]"
}
```

**Note**: `schedule_date` is required. Default to current date/time.

Present confirmation before creating:
```
## New RMA
Device:   CY-RIO — CY-RIO-29-12
Customer: AMP VISUAL TV (Sylvain Delahousse)
Fault:    Ethernet port not responding
Warranty: No → Paid repair
Source:   [support thread link]
Tech:     Paul Rathgeb

Create this RMA?
```

### 2. Check RMA status / Dashboard

**Trigger**: "what's in repair", "RMA status", "open RMAs", "repair dashboard"

#### Fetch active repairs:
```
model: repair.order
method: search_read
kwargs_json: {
  "domain": [["state", "not in", ["done", "cancel"]]],
  "fields": ["id", "name", "product_id", "lot_id", "partner_id", "state",
             "x_studio_cyanview_status", "x_studio_fault_description",
             "under_warranty", "schedule_date", "user_id", "tag_ids"],
  "order": "schedule_date asc"
}
```

Present as dashboard:
```
## Active RMAs (4 open)

| RMA | Device | Serial | Customer | Status | Fault | Days |
|-----|--------|--------|----------|--------|-------|------|
| RMA2601-032 | RCP-DUO | CY-RCP-40-174 | TV Prod | Accepted | Joystick disconnect | 3 |
| RMA2601-031 | RCP-DUO | CY-RCP-40-34 | TV Prod | Accepted | Joystick disconnect | 3 |
| RMA2601-030 | RIO | CY-RIO-29-12 | AMP VISUAL | Accepted | — | 4 |
| RMA2601-020 | RIO | CY-RIO-58-35 | DTC Ltd | Accepted | — | 23 |

⚠ RMA2601-020 has been in "Accepted" for 23 days — needs attention
```

### 3. Update RMA status

**Trigger**: "update RMA [number]", "device arrived", "repair done", "ship back"

#### Advance the Cyanview status:
```json
{
  "model": "repair.order",
  "method": "write",
  "args_json": "[[RMA_ID], {\"x_studio_cyanview_status\": \"NEW_STATUS\"}]"
}
```

#### Advance the Odoo workflow state (use action methods):

| Transition | Method |
|-----------|--------|
| draft → confirmed | `action_validate` |
| confirmed → under_repair | `action_repair_start` |
| under_repair → done | `action_repair_end` |
| any → cancel | `action_repair_cancel` |

```
model: repair.order
method: action_repair_start
args_json: "[[RMA_ID]]"
```

**Important**: The Cyanview status (`x_studio_cyanview_status`) and Odoo `state`
are semi-independent. Always update BOTH when advancing:

| Action | Cyanview status | Odoo state method |
|--------|----------------|-------------------|
| Device arrived | `2-arrived` | — |
| Fault confirmed | `3-confirmed` | `action_validate` |
| Start repair | `4-underRepair` | `action_repair_start` |
| Repair complete | `5-repaired` | `action_repair_end` |
| Shipped back | `6-returned` | — |
| Fully closed | `7-done` | — |

### 4. Add diagnostic / repair notes

**Trigger**: "add diagnostic to RMA [number]", "repair notes"

Write to `internal_notes` using structured HTML format:
```html
<h3>Diagnostic</h3>
<div>Description of findings after inspection.</div>

<h3>Repair</h3>
<div>What was done to fix the device.</div>
```

```json
{
  "model": "repair.order",
  "method": "write",
  "args_json": "[[RMA_ID], {\"internal_notes\": \"<h3>Diagnostic</h3><div>FINDINGS</div><h3>Repair</h3><div>ACTIONS TAKEN</div>\"}]"
}
```

**Warning**: This OVERWRITES existing notes. If notes already exist, READ first,
then APPEND the new content.

### 5. Search RMA by serial number

**Trigger**: "RMA history for [serial]", "has this device been repaired before"

```
model: repair.order
method: search_read
kwargs_json: {
  "domain": [["lot_id.name", "ilike", "SERIAL"]],
  "fields": ["id", "name", "product_id", "lot_id", "partner_id", "state",
             "x_studio_cyanview_status", "x_studio_fault_description",
             "create_date", "under_warranty"],
  "order": "create_date desc"
}
```

Present history:
```
## Repair history for CY-RIO-52-33

| RMA | Date | Customer | Fault | Warranty | Status |
|-----|------|----------|-------|----------|--------|
| RMA2601-028 | 2026-03 | NEP UK | Ethernet port died | No | Done |
| RMA2510-004 | 2025-10 | NEP UK | Network issue | No | Done |

⚠ Repeat offender — 2 repairs in 5 months. Consider replacement.
```

### 6. Search RMA by customer

**Trigger**: "RMAs for [customer]", "repair history for [company]"

```
model: repair.order
method: search_read
kwargs_json: {
  "domain": ["|", ["partner_id.name", "ilike", "CUSTOMER"], ["partner_id.parent_id.name", "ilike", "CUSTOMER"]],
  "fields": ["id", "name", "product_id", "lot_id", "state",
             "x_studio_cyanview_status", "x_studio_fault_description",
             "create_date", "under_warranty"],
  "order": "create_date desc",
  "limit": 20
}
```

### 7. Monthly RMA report

**Trigger**: "RMA report", "repair stats", "how many repairs this month"

Fetch all RMAs for the period and summarize:
```
model: repair.order
method: search_read
kwargs_json: {
  "domain": [["create_date", ">=", "YYYY-MM-01"]],
  "fields": ["id", "name", "product_id", "state", "under_warranty", "tag_ids",
             "x_studio_cyanview_status", "create_date"],
  "order": "create_date desc"
}
```

Present:
```
## RMA Report — March 2026

Total: 4 new RMAs
  - RIO: 2 (50%)
  - RCP-DUO: 2 (50%)

By status:
  - In progress: 4
  - Completed: 0

Warranty: 0 | Out of warranty: 4
Avg resolution time: — (none completed yet)

Trend: ↑ vs Feb (3 RMAs) — watch RCP-DUO pattern
```

## Common fault patterns (for smart diagnostics)

When a user describes a fault, suggest likely causes based on historical patterns:

| Symptom | Likely product | Common cause | Typical fix |
|---------|---------------|--------------|-------------|
| "Ethernet not working" / "no network" | RIO, VP4 | Dirty connector / firmware bug | Clean connector, update firmware |
| "Joystick disconnect" / "slow response" | RCP-DUO, RCP-J | Loose internal cable / firmware | Resolder, update firmware |
| "Screen broken" / "LCD dead" | RCP-J | Physical damage | Replace LCD |
| "Not booting" / "no power" | Any | Power supply or autoboot setting | Check PSU, enable autoboot |
| "IP settings wrong" / "DHCP not working" | VP4, RIO | Corrupted network config | Reset IP, update firmware |
| "Camera not detected" | CI0, RIO | Cable issue or serial port | Check cable, test serial port |

## Edge cases

- **Multiple devices from same customer**: Create separate RMA per serial number
  (even if same fault). Each device gets its own repair order.
- **No serial number** (e.g., cable): `lot_id` will be `false`. Set `product_id` only.
- **Device not in Odoo**: Search product by `default_code`. If truly unknown, ask user.
- **Repeat repairs**: When creating RMA, always check repair history for that serial.
  Flag if repaired more than once in 12 months.
- **DOA**: Tag as DOA + Warranty. These get priority handling.
- **Loaner device**: If customer needs a replacement while waiting, note it. Loaner
  devices go from CY/Loan (id: 44).
- **Linked helpdesk ticket**: If creating RMA from a support ticket, set `ticket_id`.
