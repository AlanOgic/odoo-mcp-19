---
name: cyanview-project-designer
description: >
  Design Cyanview camera control system architectures with Excalidraw diagrams.
  Takes broadcast/live production requirements and produces technical architecture:
  device selection, network topology, cabling, compatibility checks, and visual
  system diagram. Pre-sales technical design tool — validates before quoting.
  Triggers on: "design a system", "system architecture", "project design",
  "network diagram", "how to connect", "system diagram", "draw a system",
  "schema", "schéma", "topology", "câblage", "wiring", "setup diagram",
  "how would I connect X cameras", "what do I need for", "technical design",
  "broadcast setup", "live production setup", "studio design", "REMI setup",
  "remote production", "fly-pack design", "OB van setup", or any request to
  design or visualize a Cyanview camera control system.
allowed-tools: mcp__MCP_DOCKER__export_to_excalidraw, mcp__MCP_DOCKER__create_view, mcp__Clorag-mcp__search, mcp__Clorag-mcp__search_cameras, mcp__Clorag-mcp__get_camera
---

# Cyanview Project Designer

Design complete Cyanview camera control architectures and generate visual system diagrams.
This skill bridges the gap between "what does the customer need?" and "here's the quote" —
it's the technical pre-sales tool that validates compatibility, picks the right topology,
and draws a clear picture everyone can understand.

## Why this matters

Cyanview systems can be configured in many ways. A customer saying "I need to control 8
cameras remotely" doesn't tell you enough — are they PTZ or studio cameras? Serial or IP
control? Single operator or multi-RCP? LAN or WAN? The wrong assumptions lead to wrong
quotes, returns, and unhappy customers. This skill forces the right questions upfront,
validates the answers against real product constraints, and outputs a visual architecture
that both the customer and the Cyanview team can review before committing to a quote.

## Workflow

### Step 1: Extract project requirements

Parse the user's message for these elements. Infer what you can, ask only what's missing.

| Element | How to infer | Ask only if... |
|---------|-------------|----------------|
| **Camera count** | Number mentioned | Not mentioned |
| **Camera brands/models** | Brand or model names | Needed for cable selection |
| **Control protocol** | Serial (8P/6P/LANC) vs IP | Ambiguous — affects CI0 vs RIO choice |
| **Network type** | "remote"/"WAN"/"internet"/"REMI" → WAN; "local"/"LAN"/"studio" → LAN | Not mentioned |
| **Use case** | Studio, OB van, fly-pack, REMI, house of worship, esports, etc. | Not critical but helps design |
| **Operator count** | "1 operator"/"multi-op"/"2 RCPs" | >8 cameras or multi-location |
| **Controller preference** | RCP-J (default) vs RCP (compact) | Never ask — default RCP-J |
| **Special requirements** | Tally, GPIO, video processing, SDI I/O | Only if mentioned |
| **Existing equipment** | "we already have 2 RIOs" | Only if mentioned |

**Golden rule**: if someone says "design a system for 6 Sony FR7 cameras remote production",
you have everything — go straight to design.

### Step 2: Select architecture pattern

Read `references/architecture-patterns.md` for full details. Quick decision tree:

```
How many cameras?
├── 1-2, local, serial control
│   └── DIRECT: RCP → CI0 → Camera (no RIO, no licence on CI0)
│
├── 1-16, local (LAN)
│   └── STANDARD-LAN: RCP → Network → RIO(s) + LAN licence → Camera(s)
│
├── 1-16, remote (cloud relay)
│   └── REMOTE-CLOUD: RCP → Cloud Relay → 1× RIO + WAN licence → [venue LAN] → CI0(s)/cameras
│
├── 1-16, remote (customer VPN)
│   └── REMOTE-VPN: RCP → VPN → 1× RIO + LAN licence → [venue LAN] → CI0(s)/cameras
│
├── Mixed LAN + WAN
│   └── HYBRID: RCP → some RIOs LAN + some RIOs WAN
│
├── Multi-operator (multiple RCPs)
│   └── MULTI-RCP: Multiple RCPs on same network, each with own licence
│
├── Compact / 1-RCP-per-camera
│   └── DISTRIBUTED: N × (RCP + DUO licence) + N × RIO
│
└── Special: mini-cameras (Dreamchip, Marshall)
    └── MINI-CAM: RCP → CI0 (direct, no licence) + specific cable
```

### Step 3: Validate compatibility

Before generating the diagram, run these checks:

**Camera-cable compatibility**: For each camera brand/model, verify the correct cable exists.
Reference: the cable table in this file (same as quote builder).

| Brand | Serial Cable | IP Control | Notes |
|-------|-------------|------------|-------|
| Sony (BRC, FR7, etc.) | CY-CBL-SONY-8P-03 | Yes (IP too) | Most models support both |
| Panasonic PTZ | CY-CBL-6P-PAN-10P | Yes (IP too) | AW-UE series = IP-only |
| Canon PTZ | CY-CBL-6P-CN-REM | Yes (IP too) | CR-N series |
| Fujifilm | CY-CBL-6P-FUJI-02/03 | No | Serial only |
| Marshall | CY-CBL-6P-MARS-02 | Yes | Mini-camera, use CI0 |
| Dreamchip Atom | CY-CBL-6P-DCHIP-01 (or -03 for PT heads) | No | Mini, use CI0 |
| Dreamchip SSM500 | CY-CBL-6P-DCHIP-02 | No | Use CI0 |
| Bradley | CY-CBL-6P-BRADLEY | No | Serial only |
| B4 lens cameras | CY-CBL-6P-B4-01/02 | N/A | Lens control via B4 mount |
| LANC cameras | CY-CBL-6P-LANC-2/3 | No | Generic LANC protocol |
| Tilta Nucleus-M | CY-CBL-TILTA-SERIAL or USB | No | Lens motor, not camera |
| IOI/Flir | CY-CBL-6P-IOI/-02 | No | Industrial cameras |

**Licence validation**:
- RCP/RCP-J: licence tier must cover total camera count (DUO≤2, QUATTRO≤4, OCTO≤8, MSU≤16)
- RIO: needs LAN or WAN licence — **choice depends on connection method, not physical distance**
- CI0/CI03P/CI0BM/NIO/TALLY-BOX: no licence needed
- VP4: needs CY-LIC-VP4, limited stock 2026

**CRITICAL — RIO licence selection for remote setups**:

The WAN licence is **only needed when using the Cyanview cloud relay**. If the customer
manages their own VPN, the RIO and RCP appear on the same virtual subnet — a **LAN licence
is sufficient**, even if the RIO is physically on the other side of the world.

```
Remote via Cloud Relay → RIO needs WAN licence (CY-LIC-RIO-WAN)
Remote via customer VPN → RIO needs LAN licence only (CY-LIC-RIO-LAN) — same subnet via VPN
Local (same physical LAN) → RIO needs LAN licence (CY-LIC-RIO-LAN)
```

**Always ask**: "Will you manage your own VPN, or use Cyanview cloud relay?"
- VPN → LAN licence (cheaper)
- Cloud relay → WAN licence
- Don't know yet → quote WAN (safe default, can always downgrade)

**Network constraints — CRITICAL WAN ARCHITECTURE**:

In a remote scenario, **1 RIO is the gateway** at the distant site. The RIO bridges the
venue's local LAN to the studio RCP (via cloud relay or VPN). All cameras, CI0s, and
other devices at the venue connect to the RIO's local network — NOT each camera getting
its own RIO.

```
[STUDIO]                              [VENUE — Local LAN]
RCP-J ══Cloud/VPN══> 1× RIO (gateway) ──LAN──> CI0 #1 ──6P──> Camera #1
                                        ──LAN──> CI0 #2 ──6P──> Camera #2
                                        ──LAN──> CI0 #N ──6P──> Camera #N
                                        ──LAN──> NIO (GPIO/tally)
                                        ──6P──>  Camera (direct serial on RIO ports)
```

- The RIO gateway **imports all venue devices** to the distant RCP
- CI0s on the venue LAN connect to the RIO and are discovered by the remote RCP
- The RIO itself also has 2 serial 6P ports for direct camera connection
- Only **1 licence** is needed (on the gateway RIO) — WAN or LAN depending on connection method
- CI0s behind the RIO need no licence
- For multiple separate remote sites: 1 RIO gateway per site
- For a single venue with many cameras: 1 RIO + N CI0s (much cheaper than N RIOs)

**Other network rules**:
- RIO LAN = same subnet as RCP (physical LAN or VPN)
- CI0 is local/direct only — but CAN sit behind a RIO gateway at a remote site
- Multiple RCPs on same network: each sees all RIOs, camera assignment is software-based

**Capacity limits**:
- 1 RCP controls up to 16 cameras (MSU licence)
- 1 RIO/CI0 has 2 serial 6P ports — can sometimes control 2 cameras
- For >16 cameras: need multiple RCPs
- 1 RIO gateway can serve many CI0s/cameras on its local network

**Flag warnings** for the user:
- VP4 stock is limited in 2026
- NIO may require a licence in the future
- RIO-LIVE is legacy — recommend RIO + LAN licence
- >8 cameras remote = consider bandwidth/latency implications
- Mixed brands on same RIO = technically possible but confirm with support
- Remote: 1 RIO per site, NOT 1 RIO per camera (common mistake)
- VPN = LAN licence savings — always ask the customer about their network setup

### Step 4: Generate the system diagram

Use the `create_view` tool (Excalidraw) to draw the architecture. This is the core visual output.

#### Diagram conventions

**Color coding** (consistent across all project diagrams):

| Element | Fill Color | Stroke Color | Role |
|---------|-----------|-------------|------|
| RCP / RCP-J | `#d0bfff` (light purple) | `#8b5cf6` | Controller |
| RIO | `#a5d8ff` (light blue) | `#4a9eed` | Remote interface |
| CI0 / CI03P / CI0BM | `#c3fae8` (light teal) | `#06b6d4` | Direct interface |
| Camera | `#ffd8a8` (light orange) | `#f59e0b` | Camera |
| NIO | `#b2f2bb` (light green) | `#22c55e` | Network I/O |
| TALLY-BOX | `#fff3bf` (light yellow) | `#f59e0b` | Tally |
| VP4 | `#eebefa` (light pink) | `#ec4899` | Video processor |
| RSBM | `#ffc9c9` (light red) | `#ef4444` | SDI I/O |
| Network zone (LAN) | `#dbe4ff` opacity 30 | `#4a9eed` | LAN zone |
| Network zone (WAN) | `#e5dbff` opacity 30 | `#8b5cf6` | WAN/Internet zone |

**Layout patterns**:

- **Controller on the left**, cameras on the right
- **Network in the middle** (as a zone rectangle)
- **Stack cameras vertically** when >4, or use a 2-column grid
- **Group by brand** if mixed cameras
- Use labeled arrows for connection types: "Ethernet", "Serial 6P", "SDI", "USB", "WiFi"
- Show cable references on arrows (e.g., "CY-CBL-SONY-8P-03")

**Diagram structure** (left to right):

For LAN setups (1 RIO per camera, all on same local network):
```
[RCP-J] --Ethernet--> | LAN      | --Ethernet--> [RIO #1] --Serial 6P--> [Camera #1]
                       | Network  |               [RIO #2] --Serial 6P--> [Camera #2]
                       |          |               [RIO #N] --Serial 6P--> [Camera #N]
```

For remote setups (1 RIO gateway, CI0s behind it):
```
[STUDIO]                                    [VENUE — Local LAN]
[RCP-J] ==Cloud Relay or VPN==> [RIO (gw)] --Ethernet--> [CI0 #1] --6P--> [Camera #1]
                                             --Ethernet--> [CI0 #2] --6P--> [Camera #2]
                                             --6P direct-> [Camera #3] (via RIO's own 6P ports)
```
Cloud relay → RIO needs WAN licence | Customer VPN → RIO needs LAN licence only

For direct/CI0 setups (local, no RIO):
```
[RCP-J] --Ethernet--> [CI0] --Serial 6P--> [Camera]
```

#### Drawing workflow

1. **Camera**: Start with a `cameraUpdate` (size depends on complexity):
   - 1-4 devices: 800×600
   - 5-10 devices: 1200×900
   - 10+ devices: 1600×1200

2. **Title**: Project name or description at the top

3. **Controller zone** (left): Draw RCP(s) with licence label

4. **Network zone** (center): Semi-transparent zone rectangle labeled "LAN" or "WAN/Internet"

5. **Interface zone** (right of network): RIOs or CI0s

6. **Camera zone** (far right): Cameras with brand/model labels

7. **Connections**: Arrows with protocol labels between each layer

8. **Accessories** (bottom or side): Tally, NIO, VP4, RSBM if present

9. **Legend** (bottom-right corner): Small color key if diagram is complex

#### Example: 4 Sony FR7, remote via cloud relay

```
Title: "Remote Production — 4× Sony FR7 (Cloud Relay)"

[STUDIO]                                      [VENUE — Local LAN]
[RCP-J]  ══Cloud Relay══>  [RIO (gateway)]  --Ethernet--> [CI0 #1] → [Sony FR7 #1]
 QUATTRO                                    --Ethernet--> [CI0 #2] → [Sony FR7 #2]
                                            --Ethernet--> [CI0 #3] → [Sony FR7 #3]
                                            --Ethernet--> [CI0 #4] → [Sony FR7 #4]

Arrows:
  RCP-J → RIO: "Cloud Relay"
  RIO → each CI0: "Ethernet (venue LAN)"
  each CI0 → Camera: "CY-CBL-SONY-8P-03"

Devices: 1× RCP-J, 1× RIO + WAN licence, 4× CI0 (no licence), 4× cables
```

#### Example: same setup but customer manages VPN

```
Title: "Remote Production — 4× Sony FR7 (VPN)"

Same topology — but RIO gets LAN licence instead of WAN (VPN puts everything on same subnet)
Devices: 1× RCP-J, 1× RIO + LAN licence, 4× CI0 (no licence), 4× cables
→ Cheaper: LAN licence instead of WAN
```

#### Example: 4 Sony FR7, LAN (local production)

```
Title: "Studio — 4× Sony FR7"

[RCP-J]  --Ethernet-->  [LAN Switch]  --Ethernet--> [RIO #1 (LAN)] → [Sony FR7 #1]
 QUATTRO                               --Ethernet--> [RIO #2 (LAN)] → [Sony FR7 #2]
                                        --Ethernet--> [RIO #3 (LAN)] → [Sony FR7 #3]
                                        --Ethernet--> [RIO #4 (LAN)] → [Sony FR7 #4]

Devices: 1× RCP-J, 4× RIO + LAN licence each, 4× cables
```

### Step 5: Present the summary

After the diagram, present a text summary:

**Architecture summary format:**

```
## System Architecture: [Project Name]

**Pattern**: [STANDARD-WAN / STANDARD-LAN / DIRECT / HYBRID / MULTI-RCP / DISTRIBUTED]
**Cameras**: [count] × [brand/model]
**Network**: [LAN / WAN / Hybrid]
**Operators**: [count]

### Bill of Materials

| Device | Qty | Notes |
|--------|-----|-------|
| CY-RCP-J | 1 | Controller (studio side) |
| CY-LIC-RCP-QUATTRO | 1 | 4-camera licence |
| CY-RIO | 1 | Gateway (venue side) |
| CY-LIC-RIO-WAN or LAN | 1 | WAN if cloud relay / LAN if customer VPN |
| CY-CI0 | 4 | Camera interface (venue LAN, no licence) |
| CY-CBL-SONY-8P-03 | 4 | Serial cable per camera |

### Compatibility Notes
- [any warnings or special considerations]

### Scalability
- Current: 4 cameras (QUATTRO)
- Upgrade path: QUATTRO → OCTO (up to 8 cameras) via CY-LIC-RCP-4-TO-8
- Max: MSU licence supports up to 16 cameras

### Network Requirements
- [bandwidth, latency, port forwarding, VPN notes for WAN setups]
```

### Step 6: Offer next steps

After presenting the design:
1. **"Want me to create a quote?"** — hand off to cyanview-quote skill
2. **"Want me to check stock?"** — hand off to cyanview-inventory-watchdog skill
3. **"Want a customer briefing first?"** — hand off to cyanview-customer-360 skill
4. **"Want to export the diagram?"** — use `export_to_excalidraw` for a shareable link

## Advanced scenarios

### REMI / At-Home Production
Remote production where cameras are at the venue and operators are in the studio.
- **1 RIO per remote site** — the RIO is the venue gateway
- All cameras at the venue connect via CI0s on the venue's local LAN (behind the RIO)
- The RIO's own 2 serial 6P ports can also connect cameras directly
- CI0s behind the RIO need no licence — only the gateway RIO needs a licence
- For multiple separate venues: 1 RIO gateway per venue
- **Licence depends on connection method**:
  - Customer manages VPN (WireGuard, OpenVPN, etc.) → **LAN licence** (RIO on same virtual subnet as RCP)
  - No VPN, uses Cyanview cloud relay → **WAN licence**
  - Always ask: "Do you have a VPN between studio and venue?"
- Consider bandwidth: each RIO control stream is lightweight (~50kbps) but video is separate
- Latency: Cyanview control works up to ~200ms RTT, beyond that manual iris/focus becomes difficult
- Recommend: dedicated management VLAN for Cyanview traffic
- This architecture is much more cost-effective than 1 RIO per camera

### Multi-RCP / Multi-Operator
Multiple operators controlling different camera subsets:
- Each RCP needs its own licence tier (based on cameras it will control)
- All RCPs and RIOs must be on the same network
- Camera assignment is done in Cyanview software (not hardware)
- Example: 12 cameras, 2 operators → RCP-J #1 (OCTO, cams 1-8) + RCP-J #2 (QUATTRO, cams 9-12)

### Fly-Pack / Portable
Compact, portable systems for events:
- Prefer CY-RCP (compact, no joystick) to save space
- CI0 instead of RIO if everything is local (no licence cost)
- Consider CY-MEC-RCP-FRAME for rack mounting
- Power: CY-PWR-1A-21 for each CI0/RIO, or D-Tap (CY-CBL-DTAP) from camera battery

### House of Worship / Fixed Install
Permanent installations:
- LAN is typical (everything on-site)
- Consider extended warranties (CY-WAR-EXT-2Y / CY-WAR-EXT-3Y)
- Tally integration (CY-TALLY-BOX + CY-CBL-6P-TALLY)
- GPIO integration (NIO for automation triggers)

### Esports / Multi-Camera Switching
High camera count, fast switching:
- VP4 for video processing (if in stock)
- RSBM for SDI I/O with Blackmagic switchers
- NIO for GPIO tally integration with production switcher
- Consider MSU licence from the start for scalability

### Mixed Camera Brands
When a setup includes cameras from different manufacturers:
- Each brand may need different cables
- Group by brand in the diagram
- Some brands are IP-only (no serial cable needed)
- All cameras appear in the same Cyanview software regardless of brand

## Edge cases and warnings

- **Camera with both serial and IP**: ask user which control mode they prefer. IP = no cable, serial = cable needed
- **>16 cameras**: must use multiple RCPs, each with its own licence
- **CI0 over WAN**: CI0 cannot connect directly over WAN, but CAN sit behind a RIO WAN gateway on the venue LAN. The RIO WAN bridges CI0s to the remote RCP
- **2 cameras per RIO/CI0**: depends on protocol — some cameras use both 6P ports, some only one. When in doubt, quote 1 RIO per camera
- **WiFi control**: possible via MIS-ALFA-USB-HALOW or MIS-USB-WIFI-GEN on RIO USB port. Not recommended for mission-critical
- **Blackmagic cameras**: use CI0BM variant (has SDI I/O) or RIO + RSBM
- **Lens-only control**: Tilta Nucleus-M via CY-CBL-TILTA-SERIAL or CY-CBL-TILTA-USB — this controls the lens motor, not the camera body
