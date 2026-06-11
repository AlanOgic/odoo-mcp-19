---
name: cyanview-quote
description: >
  Build Cyanview sales quotations in Odoo. Triggers on: "devis", "quote", "quotation", "SO",
  "sales order", "commande", "price list", "pricing", "how much for", "quote for X cameras",
  "configure a system", "how much would it cost", "price for", "prepare an offer",
  "build a system for [company]", or any request involving CY- product selection and pricing
  — even casual like "4 Sony cameras remote for ProMedia" or "what's the price for 2 RIOs".
allowed-tools: mcp__odoo19-mcp__execute_method, mcp__odoo19-mcp__batch_execute, mcp__odoo19-mcp__read_resource
argument-hint: "[customer] [cameras] [details]"
---

# Cyanview Quote Builder

## Quick-start: what to extract from the user's request

Parse the user's message for these elements. Only ask about what's missing — infer the rest.

| Element | How to infer | Ask only if… |
|---------|-------------|--------------|
| **Customer** | Company name mentioned | Not mentioned at all |
| **Qty cameras** | Number mentioned ("4 cameras", "a pair") | Ambiguous |
| **Camera brand** | Brand or model mentioned (PXW-Z450=Sony, HPX500=Panasonic) | Not mentioned AND cables needed |
| **Network** | "remote"/"WAN"/"internet" → WAN; "local"/"LAN"/"studio" → LAN | Not mentioned |
| **Controller** | Default: RCP-J. User says "RCP"/"compact"/"mini" → CY-RCP | Never ask — default to RCP-J |
| **RCP topology** | Default: 1 RCP for system. User says "1 per camera"/"compact" → 1 RCP each with DUO | Only if ambiguous on large setups |

**Golden rule**: if the user says "4 Sony cameras remote for CompanyX", you have everything —
go straight to building the quote, no questions needed.

## Business rules (quick reference)

Full details: see Appendix A (Product Catalog) and Appendix B (Licence Matrix) at the end of this document.

### Licence selection (mandatory — device won't work without it)

```
RCP/RCP-J → pick by camera count:
  1-2 cam  → CY-LIC-RCP-DUO
  3-4 cam  → CY-LIC-RCP-QUATTRO
  5-8 cam  → CY-LIC-RCP-OCTO
  9-16 cam → CY-LIC-RCP-MSU

  If topology = 1 RCP per camera → each RCP gets CY-LIC-RCP-DUO

RIO → pick by network:
  LAN/local  → CY-LIC-RIO-LAN
  WAN/remote → CY-LIC-RIO-WAN

VP4 → CY-LIC-VP4 (check stock — limited availability 2026)

CI0/CI03P/CI0BM/NIO/TALLY-BOX → no licence needed
```

### Legacy — Do Not Quote

These products exist in Odoo but must NEVER appear on new quotes:
- **RCP-DUO / RCP-DUO-J / RCP-QUATTRO / RCP-OCTO-J** — bundled device+licence. Always quote device + licence separately.
- **RIO-LIVE** — discontinued. Replaced by RIO + CY-LIC-RIO-LAN. If customer mentions it, quote RIO + LAN licence.
- **GWY** — discontinued.

### Camera adaptor cables (1 per camera — serial control only)

Adaptor cables are ONLY needed for cameras controlled via a serial port (8-pin, LANC, 10-pin, remote, etc.).
**IP-controlled cameras do NOT need a cable** — the RIO/CI0 communicates via network.

If the user mentions a camera model, determine the control protocol:
- **Serial** → quote the matching CY-CBL cable
- **IP** → no cable needed
- **Both possible** → ask the user which control mode they're using

| Brand | Model examples | Cable |
|-------|---------------|-------|
| Sony | BRC-X1000, BRC-H900 | CY-CBL-SONY-8P-03 |
| Panasonic | AJ-HPX2700, AG-HPX500 | CY-CBL-6P-PAN-10P |
| Canon | CR-N700, CR-N500, CR-X500 | CY-CBL-6P-CN-REM |
| Fujifilm | SX800 | CY-CBL-6P-FUJI-02 / -03 |
| Marshall | CV630-IP | CY-CBL-6P-MARS-02 |
| Dreamchip | Atom (all models) | CY-CBL-6P-DCHIP-01 (default) or -03 (thin, for PT heads) |
| Dreamchip | SSM500 | CY-CBL-6P-DCHIP-02 |
| IOI/Flir | — | CY-CBL-6P-IOI / -02 |
| Bradley | — | CY-CBL-6P-BRADLEY |
| Tilta | Nucleus-M (serial) | CY-CBL-TILTA-SERIAL |
| Tilta | Nucleus-M (USB) | CY-CBL-TILTA-USB |
| B4 mount | Any B4 lens | CY-CBL-6P-B4-01 / -02 |
| LANC | Any LANC camera | CY-CBL-6P-LANC-2 / -3 |

Note: CI0/RIO have 2 serial 6P ports — depending on protocol, 1 unit can sometimes control 2 cameras. In that case, quote 1 cable per camera but possibly fewer CI0/RIO units. Ask user if unclear.

### Accessories — only on explicit request

Do NOT auto-suggest. Only add if the user asks.
Common ones: CY-PWR-1A-21 (PSU), CY-MEC-RCP-FRAME (rack), CY-TALLY-BOX,
CY-CBL-JACK-GPIO8, CY-CBL-DTAP, MIS-LAN-USB-DONGLE-FX6, warranties (CY-WAR-EXT-2Y/3Y).

**RSBM**: quote only the RSBM device itself. The included CY-CBL-6P-ST-15 cable is bundled — do not quote it separately.

### Upgrade paths (mention at end of quote)

Upgrades exist: DUO→QUATTRO, DUO→OCTO, DUO→MSU, QUATTRO→OCTO, QUATTRO→MSU, OCTO→MSU, LAN→WAN.
Always mention as a scalability selling point.

## Quote structure

### Grouping: identical kits share one section

Group devices that have the **same licence type** (and same cable) into a single section.
Split only when licence type, camera brand, or device type differs.

**Section name format**: `"Nx [Device] [Licence-tier]"` (drop "1x" for single devices)

Examples: `"4x RIO WAN — Sony"`, `"RCP-J QUATTRO"`, `"4x RCP DUO"`, `"2x RIO LAN — Panasonic"`

### Reference examples

**Simple: "4 Sony cameras, remote"**
```
Section: RCP-J QUATTRO
  CY-RCP-J              ×1
  CY-LIC-RCP-QUATTRO    ×1

Section: 4x RIO WAN — Sony
  CY-RIO                ×4
  CY-LIC-RIO-WAN        ×4
  CY-CBL-SONY-8P-03     ×4
```

**Mixed: "4 Sony remote + 2 Panasonic local"**
```
Section: RCP-J OCTO
  CY-RCP-J              ×1
  CY-LIC-RCP-OCTO       ×1

Section: 4x RIO WAN — Sony
  CY-RIO                ×4
  CY-LIC-RIO-WAN        ×4
  CY-CBL-SONY-8P-03     ×4

Section: 2x RIO LAN — Panasonic
  CY-RIO                ×2
  CY-LIC-RIO-LAN        ×2
  CY-CBL-6P-PAN-10P     ×2
```

**Compact: "4 cameras, 1 RCP per camera"**
```
Section: 4x RCP DUO
  CY-RCP                ×4
  CY-LIC-RCP-DUO        ×4

Section: 4x RIO LAN — Sony
  CY-RIO                ×4
  CY-LIC-RIO-LAN        ×4
  CY-CBL-SONY-8P-03     ×4
```

**Direct connect: "2 Sony, local, no RIO"**
```
Section: RCP-J DUO
  CY-RCP-J              ×1
  CY-LIC-RCP-DUO        ×1

Section: 2x CI0 — Sony
  CY-CI0                ×2
  CY-CBL-SONY-8P-03     ×2
```
(CI0 has no licence — it connects directly to the RCP)

**US customer: "4 Sony cameras, remote, for a US client"**
```
Section: RCP-J QUATTRO
  CY-RCP-J              ×1
  CY-LIC-RCP-QUATTRO    ×1

Section: 4x RIO WAN — Sony
  CY-RIO                ×4
  CY-LIC-RIO-WAN        ×4
  CY-CBL-SONY-8P-03     ×4

Section: Import Duties & Taxes (DDP)
  XTR-PREP-DUTIES-TAXES ×1  (price_unit = 15% of physical goods subtotal + 80)
```
(Duties section is always last; incoterm set to DDP on the SO)

## Create in Odoo

### Pricing: automatic via pricelist

**Do NOT set `price_unit` on SO lines.** Odoo automatically applies the correct pricelist
based on the customer's partner record (USD for US clients, EUR for others, etc.).
Just set `product_id` and `product_uom_qty` — Odoo computes the price via onchange.

**Exception**: `price_unit` IS manually set on the XTR-PREP-DUTIES-TAXES line (see US duties rule below).

### US clients: Duties & Taxes + Incoterm DDP

**When the customer's `country_id` = United States**, apply these extra steps:

1. **Add XTR-PREP-DUTIES-TAXES in its own section** at the end of the quote:
   - First create a **section header** line (`display_type: 'line_section'`, name: `"Import Duties & Taxes (DDP)"`) at the next sequence number
   - Then create the XTR-PREP-DUTIES-TAXES product line under that section
   - Read back all SO lines after creation to get their `price_subtotal` (post-pricelist, post-discount)
   - Sum only **physical goods** lines: devices (CY-DEV-*), cables (CY-CBL-*), accessories (CY-PWR-*, CY-MEC-*), third-party (MIS-*)
   - **Exclude** licences (CY-LIC-*) from the sum — they are not physical goods
   - Formula: `price_unit = SUM(physical lines price_subtotal) × 0.15 + 80`
   - Force `price_unit` manually on this line (exception to the "do not set price_unit" rule)

2. **Set Incoterm on the SO** to **DDP**:
   ```
   model: sale.order
   method: write
   args: [[SO_ID], {"incoterm": <DDP_ID>, "incoterm_location": "<delivery country>"}]
   ```
   - Look up DDP incoterm: `search_read` on `account.incoterms` with `[["code", "=", "DDP"]]`
   - `incoterm_location` = country name from the customer's delivery address

### Step 1: Lookup products (one batch call)

Search all needed `product.product` by `default_code` in a single `search_read`:
```
model: product.product
method: search_read
domain: [["default_code", "in", ["CY-DEV-RCP-J", "CY-DEV-RIO", "CY-LIC-RCP-QUATTRO", ...]]]
fields: ["id", "default_code"]
```

### Step 2: Find or create customer

**Search strategy** — search across multiple fields to maximize matches:
```
model: res.partner
method: search_read
domain: ["|", "|", "|",
  ["name", "ilike", "search term"],
  ["email", "ilike", "search term"],
  ["ref", "ilike", "search term"],
  ["email", "ilike", "@domain.com"]]
fields: ["id", "name", "email", "ref", "is_company", "parent_id", "country_id", "property_product_pricelist"]
```

**Resolution logic:**
- If user gives a **company name** → search company (`is_company=True`), list contacts inside
- If user gives a **person name** → search contact, get parent company via `parent_id`
- If **multiple matches** → present the list to user and ask which one
- Always check `country_id` — needed for US duties rule (see below)

**Creation** (only after thorough search + user confirmation):
1. **Ask the user** before creating: "Customer not found — shall I create it?"
2. Create the **company first** (`is_company=True`): `name`, `country_id`, + any info available (email, street, city, zip, phone, website)
3. Then create the **contact inside** (`parent_id=company_id`, `is_company=False`): name, email, phone, job title if known

### Step 3: Create SO, then batch all lines

First create the SO to get its ID:
```
model: sale.order
method: create
args: [{"partner_id": ID}]
```

Then batch all lines with `batch_execute`:
```json
[
  {"model": "sale.order.line", "method": "create",
   "args_json": "[{\"order_id\": SO_ID, \"display_type\": \"line_section\", \"name\": \"RCP-J QUATTRO\", \"sequence\": 10}]"},

  {"model": "sale.order.line", "method": "create",
   "args_json": "[{\"order_id\": SO_ID, \"product_id\": PID, \"product_uom_qty\": 1, \"sequence\": 20}]"},
  ...
]
```

### Odoo field reference

| Field | Value | Purpose |
|-------|-------|---------|
| `display_type` | `'line_section'` | Section header |
| `display_type` | `'line_note'` | Note line |
| `display_type` | `False` | Regular product line |
| `sequence` | increment by 10 | Line ordering |
| `product_id` | `product.product` ID | Always variant, not template |
| `product_uom_qty` | integer | Quantity |
| `price_unit` | **DO NOT SET** (except XTR-PREP-DUTIES-TAXES) | Auto from pricelist |

### Step 4: Verify and summarize

After creation, read back the SO to confirm:
```
model: sale.order
method: search_read
domain: [["id", "=", SO_ID]]
fields: ["name", "amount_untaxed", "amount_total", "state"]
```

Present to user:
- SO reference + link: `https://<domain>/odoo/sales/<ID>`
- Total HT and TTC (auto-calculated by Odoo from pricelist)
- Upgrade path reminder (e.g. "QUATTRO → OCTO if you grow to 8 cameras")
- Quote stays draft — user confirms when ready

## Edge cases

- **Mix LAN + WAN**: split into separate RIO sections by licence type
- **RCP vs RCP-J**: default to RCP-J, use RCP only if user says "sans joystick" / "compact" / "mini"
- **1 RCP per camera**: each gets DUO licence minimum, use CY-RCP (compact) by default for this topology
- **CI0/CI03P/CI0BM**: camera interfaces that connect directly to RCP via Ethernet (no licence needed)
- **NIO**: network I/O with 16 GPIO + 2 USB, no licence (WAN-capable, may require licence in future)
- **RSBM**: SDI I/O accessory — quote RSBM only, included CY-CBL-6P-ST-15 cable is NOT quoted separately
- **VP4**: check stock availability (limited 2026, new version in dev)
- **RIO-LIVE**: legacy — quote RIO + CY-LIC-RIO-LAN instead
- **US customer**: auto-add section "Import Duties & Taxes (DDP)" with XTR-PREP-DUTIES-TAXES line (15% of physical goods + 80 flat) + set incoterm DDP
- **Discount**: use product XTR-DISC-EXCP with negative price
- **Shipping**: use SH-SER-SHIPPING with cost as price
- **Dreamchip, Marshall (mini-cameras)**: always use CY-RCP (compact) + CI0 (direct, no licence). Cable: DCHIP-01 default, DCHIP-03 for PT heads, DCHIP-02 for SSM500
- **2 cameras per CI0/RIO**: possible depending on protocol — quote fewer units but same number of cables
- **Multiple RCPs**: each RCP needs its own licence

## Appendix A — Product Catalog

# Cyanview Product Catalog — Active Sales Products

Last updated: 2026-03-20
Source: Odoo product.template, filtered on active + sale_ok + confirmed sales history
Prices: managed by Odoo pricelists (auto-selected per customer) — NOT hardcoded here.

## Devices (CY-DEV-*) — Active

| Product | Internal Code | Notes |
|---------|--------------|-------|
| CY-RCP-J | CY-DEV-RCP-J | Controller with joystick (top seller) |
| CY-RCP | CY-DEV-RCP | Controller without joystick, compact/mini/budget |
| CY-RIO | CY-DEV-RIO | Remote camera interface — needs LAN/WAN licence |
| CY-CI0 | CY-DEV-CI0 | Direct camera interface — no licence needed |
| CY-CI03P | CY-DEV-CI03P | CI0 variant, 3-pin Hirose, 3rd digital port |
| CY-CI0BM | CY-DEV-CI0BM | CI0 variant with SDI I/O for Blackmagic |
| CY-NIO | CY-DEV-NIO | Network I/O: 16 GPIO + 2 USB (no licence) |
| CY-TALLY-BOX | CY-DEV-TALLY-BOX | Tally light controller |
| CY-RSBM | CY-DEV-RSBM | SDI I/O (includes CY-CBL-6P-ST-15, do not quote cable) |
| CY-VP4 | CY-DEV-VP4 | Video processor — limited stock 2026, new version in dev |
| CY-JST-IRIS-KIT | CY-DEV-JST-IRIS-KIT | Joystick iris kit |
| CY-RCP-X | CY-DEV-RCP-X | |
| CY-RIO-AIR | CY-DEV-RIO-AIR | |
| CY-RCP-BASE | CY-DEV-RCP-BASE | |

## Devices — Legacy (DO NOT QUOTE)

| Product | Internal Code | Replacement |
|---------|--------------|-------------|
| CY-RCP-DUO-J | CY-DEV-RCP-DUO-J | → CY-RCP-J + CY-LIC-RCP-DUO |
| CY-RCP-DUO | CY-DEV-RCP-DUO | → CY-RCP + CY-LIC-RCP-DUO |
| CY-RCP-QUATTRO | CY-DEV-RCP-QUATTRO | → CY-RCP + CY-LIC-RCP-QUATTRO |
| CY-RCP-OCTO-J | CY-DEV-RCP-OCTO-J | → CY-RCP-J + CY-LIC-RCP-OCTO |
| CY-RIO-LIVE | CY-DEV-RIO-LIVE | → CY-RIO + CY-LIC-RIO-LAN |
| CY-GWY | CY-DEV-GWY | Discontinued — do not quote |

## Cables & Adaptors (CY-CBL-*)

| Product | Code | Use Case |
|---------|------|----------|
| CY-CBL-SONY-8P-03 | CY-CBL-SONY-8P-03 | Sony cameras (FR7, BRC, etc.) |
| CY-CBL-6P-TALLY | CY-CBL-6P-TALLY | Tally connection |
| CY-CBL-JACK-GPIO8 | CY-CBL-JACK-GPIO8 | GPIO/Tally 8-channel |
| CY-CBL-DTAP | CY-CBL-DTAP | D-Tap power |
| CY-CBL-6P-B4-01 | CY-CBL-6P-B4-01 | B4 lens mount cameras |
| CY-CBL-6P-B4-02 | CY-CBL-6P-B4-02 | B4 lens mount v2 |
| CY-CBL-6P-IOI | CY-CBL-6P-IOI | IOI/Flir cameras |
| CY-CBL-6P-IOI-02 | CY-CBL-6P-IOI-02 | IOI/Flir cameras v2 |
| CY-CBL-TILTA-SERIAL | CY-CBL-TILTA-SERIAL | Tilta Nucleus (serial) |
| CY-CBL-TILTA-USB | CY-CBL-TILTA-USB | Tilta Nucleus (USB) |
| CY-CBL-6P-MARS-02 | CY-CBL-6P-MARS-02 | Marshall cameras |
| CY-CBL-6P-FUJI-02 | CY-CBL-6P-FUJI-02 | Fujifilm cameras |
| CY-CBL-6P-FUJI-03 | CY-CBL-6P-FUJI-03 | Fujifilm cameras v2 |
| CY-CBL-6P-EXT50 | CY-CBL-6P-EXT50 | Extension 50cm |
| CY-CBL-6P-EXT100 | CY-CBL-6P-EXT100 | Extension 100cm |
| CY-CBL-6P-EXT300 | CY-CBL-6P-EXT300 | Extension 300cm |
| CY-CBL-6P-EXT500 | CY-CBL-6P-EXT500 | Extension 500cm |
| CY-CBL-6P-EXT1000 | CY-CBL-6P-EXT1000 | Extension 1000cm |
| CY-CBL-6P-DCHIP-01 | CY-CBL-6P-DCHIP-01 | Dreamchip Atom (default, large connector) |
| CY-CBL-6P-DCHIP-02 | CY-CBL-6P-DCHIP-02 | Dreamchip SSM500 |
| CY-CBL-6P-DCHIP-03 | CY-CBL-6P-DCHIP-03 | Dreamchip Atom (thin, for PT heads) |
| CY-CBL-6P-LANC-2 | CY-CBL-6P-LANC-2 | LANC protocol cameras |
| CY-CBL-6P-LANC-3 | CY-CBL-6P-LANC-3 | LANC protocol cameras v2 |
| CY-CBL-6P-PAN-10P | CY-CBL-6P-PAN-10P | Panasonic PTZ cameras |
| CY-CBL-6P-CN-REM | CY-CBL-6P-CN-REM | Canon remote |
| CY-CBL-6P-FAN100 | CY-CBL-6P-FAN100 | Fan-out 100cm |
| CY-CBL-6P-PFAN | CY-CBL-6P-PFAN | Power fan-out |
| CY-CBL-6P-PWR | CY-CBL-6P-PWR | Power cable 6P |
| CY-CBL-6P-ST-15 | CY-CBL-6P-ST-15 | Straight 15cm |
| CY-CBL-6P-ST-50 | CY-CBL-6P-ST-50 | Straight 50cm |
| CY-CBL-6P-BRADLEY | CY-CBL-6P-BRADLEY | Bradley cameras |
| CY-CBL-6P-AJA-01 | CY-CBL-6P-AJA-01 | AJA devices |
| CY-CBL-SDI-D1023-F | CY-CBL-SDI-D1023-F | SDI female |
| CY-CBL-SDI-D1023-M | CY-CBL-SDI-D1023-M | SDI male |

## Licences (CY-LIC-*)

### Base licences (for new devices)

| Product | Code | For Device |
|---------|------|------------|
| CY-LIC-RCP-DUO | CY-LIC-RCP-DUO | RCP/RCP-J (2 cam) |
| CY-LIC-RCP-QUATTRO | CY-LIC-RCP-QUATTRO | RCP/RCP-J (4 cam) |
| CY-LIC-RCP-OCTO | CY-LIC-RCP-OCTO | RCP/RCP-J (8 cam) |
| CY-LIC-RCP-MSU | CY-LIC-RCP-MSU | RCP/RCP-J (16 cam) |
| CY-LIC-RIO-LAN | CY-LIC-RIO-LAN | RIO (LAN) |
| CY-LIC-RIO-WAN | CY-LIC-RIO-WAN | RIO (WAN) |
| CY-LIC-VP4 | CY-LIC-VP4 | VP4 |

### Upgrade licences

| Product | Code | Upgrade Path |
|---------|------|-------------|
| CY-LIC-RCP-2-TO-4 | CY-LIC-RCP-2-TO-4 | DUO → QUATTRO |
| CY-LIC-RCP-2-TO-8 | CY-LIC-RCP-2-TO-8 | DUO → OCTO |
| CY-LIC-RCP-2-TO-MSU | CY-LIC-RCP-2-TO-MSU | DUO → MSU |
| CY-LIC-RCP-4-TO-8 | CY-LIC-RCP-4-TO-8 | QUATTRO → OCTO |
| CY-LIC-RCP-4-TO-MSU | CY-LIC-RCP-4-TO-MSU | QUATTRO → MSU |
| CY-LIC-RCP-8-TO-MSU | CY-LIC-RCP-8-TO-MSU | OCTO → MSU |
| CY-LIC-LAN-TO-WAN | CY-LIC-LAN-TO-WAN | RIO LAN → WAN |

## Accessories & Power

| Product | Code | Use |
|---------|------|-----|
| CY-PWR-1A-21 | CY-PWR-1A-21 | PSU for CI0/NIO/RIO |
| CY-PWR-VP4 | CY-PWR-VP4 | PSU for VP4 |
| CY-MEC-RCP-FRAME | CY-MEC-RCP-FRAME | Rack frame for RCP |
| CY-RACKSHELF | CY-MEC-RACKSHELF | Rack shelf |
| CY-WKTU-RMA | CY-WKTU-RMA | RMA handling fee |
| CY-WAR-EXT-2Y | CY-WAR-EXT-2Y | +2 year warranty |
| CY-WAR-EXT-3Y | CY-WAR-EXT-3Y | +3 year warranty |

## Third-party Merchandise (MIS-*)

| Product | Code | Use |
|---------|------|-----|
| MIS-LAN-USB-DONGLE-FX6 | MIS-LAN-USB-DONGLE | USB-to-Ethernet dongle |
| MIS-ALFA-TUBE-AH | MIS-ALFA-TUBE-AH | WiFi HaLow antenna |
| MIS-ALFA-USB-HALOW | MIS-ALFA-USB-HALOW | WiFi HaLow USB adapter |
| MIS-USB-WIFI-GEN | MIS-USB-WIFI-GEN | Generic WiFi USB |

## Discount & Extra Line Items

| Product | Code | Use |
|---------|------|-----|
| XTR-DISC-EXCP | XTR-DISC-EXCP | Exceptional discount (set negative price) |
| XTR-DISC-REPL | XTR-DISC-REPL | Replacement discount |
| XTR-PREP-DUTIES | XTR-PREP-DUTIES-TAXES | Import duties/taxes |
| DHL Express | SH-SER-SHIPPING | Shipping line item |

## Appendix B — Licence Matrix

# Licence Matrix — Which Device Needs What

Prices: managed by Odoo pricelists (auto-selected per customer) — NOT hardcoded here.

## Decision tree

```
Is it an RCP or RCP-J?
├─ Yes → How many cameras will it control?
│        ├─ 1-2  → CY-LIC-RCP-DUO
│        ├─ 3-4  → CY-LIC-RCP-QUATTRO
│        ├─ 5-8  → CY-LIC-RCP-OCTO
│        └─ 9-16 → CY-LIC-RCP-MSU
│
Is it a RIO?
├─ Yes → What network type?
│        ├─ Local (same network) → CY-LIC-RIO-LAN
│        └─ Remote (internet/VPN) → CY-LIC-RIO-WAN
│
Is it a VP4?
├─ Yes → CY-LIC-VP4 (check stock — no more stock in 2026, new version in dev)
│
Is it a CI0 / CI03P / CI0BM / NIO / RSBM / TALLY-BOX?
└─ No licence needed — these are controlled by the RCP
```

Note: NIO is WAN-capable and may require a licence in the future, but not today.

## Device roles explained

Understanding the system architecture helps build correct quotes:

### Controller (1 per system OR 1 per camera)

- **RCP-J** — Main controller with joystick. Most common choice.
- **RCP** — Main controller without joystick. Compact version, mini cameras, or budget option.

The RCP controls all cameras in the system. The licence on the RCP determines
the maximum number of cameras. Two standard configurations:
- **1 RCP for the whole system** — most common, licence tier matches total camera count
- **1 RCP per camera** — for compact/distributed setups, each RCP gets a DUO licence minimum. Depends on user preference for larger setups.

### Camera Interface (1 per camera — sometimes controls 2 cameras)

Each CI0/RIO has 2 serial 6P connections. Depending on the camera protocol,
a single CI0/RIO can sometimes control 2 cameras via its 2 ports.

- **RIO** — Remote camera interface. Connects via network/serial 6P/USB to the camera, and via LAN/WAN (Internet) to the RCP. Needs its own licence (LAN or WAN).
- **CI0** — Direct camera interface. Connects via Ethernet to the RCP and via 6P cable to camera. No separate licence needed.
- **CI03P** — CI0 variant with 3-pin Hirose connector. The 3rd port is digital for specific use.
- **CI0BM** — CI0 variant which adds SDI I/O for Blackmagic cameras.
- **NIO** — Network I/O interface. Like RIO but serial 6P ports are replaced by 16 GPIO ports and 2 USB. No licence needed (WAN-capable, may require licence in the future).

### Accessories (shared across system)

- **TALLY-BOX** — Tally light controller
- **TALLY-CBL** — Tally cable for Tally-Box
- **RSBM** — SDI I/O module. Connects with included CY-CBL-6P-ST-15 to CI0/RIO, connects with SDI to Blackmagic. The CY-CBL-6P-ST-15 is included — do not quote it separately. Only RSBM itself goes on the quote.
- **VP4** — Video processor (needs CY-LIC-VP4). No more stock in 2026, new version in development — check availability before quoting.

### Legacy — Do Not Quote

These products exist in Odoo but should NOT be used on new quotes:

- **RCP-DUO / RCP-DUO-J / RCP-QUATTRO / RCP-OCTO-J** — Pre-bundled RCP + licence. Always quote device + licence separately instead.
- **RIO-LIVE** — Former version of RIO + LAN. Replaced by RIO device + CY-LIC-RIO-LAN licence. If a customer mentions RIO-LIVE, quote RIO + LAN licence instead.
- **GWY** — Gateway. Discontinued, do not quote.

## Common system configurations (quote section format)

### Small setup (2 cameras, local)
```
Section: RCP-J DUO
  CY-RCP-J              ×1
  CY-LIC-RCP-DUO        ×1

Section: 2x RIO LAN
  CY-RIO                ×2
  CY-LIC-RIO-LAN        ×2
  CY-CBL-[BRAND]-*      ×2
```

### Medium setup (4 cameras, local)
```
Section: RCP-J QUATTRO
  CY-RCP-J              ×1
  CY-LIC-RCP-QUATTRO    ×1

Section: 4x RIO LAN
  CY-RIO                ×4
  CY-LIC-RIO-LAN        ×4
  CY-CBL-[BRAND]-*      ×4
```

### Compact setup (4 cameras, 1 RCP per camera)
```
Section: 4x RCP DUO
  CY-RCP                ×4
  CY-LIC-RCP-DUO        ×4

Section: 4x RIO LAN
  CY-RIO                ×4
  CY-LIC-RIO-LAN        ×4
  CY-CBL-[BRAND]-*      ×4
```

### Large setup (8 cameras, remote/WAN)
```
Section: RCP-J OCTO
  CY-RCP-J              ×1
  CY-LIC-RCP-OCTO       ×1

Section: 8x RIO WAN
  CY-RIO                ×8
  CY-LIC-RIO-WAN        ×8
  CY-CBL-[BRAND]-*      ×8
```

## Upgrade paths

When a customer grows their system, they buy an upgrade licence instead of a new base licence.

| From | To | Upgrade Product |
|------|----|----------------|
| DUO (2) | QUATTRO (4) | CY-LIC-RCP-2-TO-4 |
| DUO (2) | OCTO (8) | CY-LIC-RCP-2-TO-8 |
| DUO (2) | MSU (16) | CY-LIC-RCP-2-TO-MSU |
| QUATTRO (4) | OCTO (8) | CY-LIC-RCP-4-TO-8 |
| QUATTRO (4) | MSU (16) | CY-LIC-RCP-4-TO-MSU |
| OCTO (8) | MSU (16) | CY-LIC-RCP-8-TO-MSU |
| RIO LAN | RIO WAN | CY-LIC-LAN-TO-WAN |

Always mention upgrade paths when quoting — it's a selling point that the system is scalable.
