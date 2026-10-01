import { describe, it, expect, beforeEach } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

// Regression test for a parser bug: a finding whose description starts on
// the line AFTER its "**N. Title**" heading (rather than inline on the same
// line) used to be stored with an empty body/recommendation, because the
// lookahead's end-of-input branch was a bare `$`, which under the /m flag
// also matches the end of the title's own line.
const REPORT = `# Test Audit

## Executive Summary

Summary text.

## Findings

### P1 — High Business Impact

**1. Multi-line finding has no body**
This description is on its own line below the title.
- **Recommendation:** Fix the thing described above.

### P2 — Meaningful Improvement

**2. Inline finding still works** This description stays on the same line as the title.
- **Recommendation:** Fix this other thing too.

---

## Ranked Summary

Nothing else here.
`;

describe("importNewAuditReports", () => {
  beforeEach(() => {
    const dataDir = mkdtempSync(path.join(tmpdir(), "audit-import-test-data-"));
    const auditDir = mkdtempSync(path.join(tmpdir(), "audit-import-test-reports-"));
    mkdirSync(path.join(auditDir, "reports"), { recursive: true });
    writeFileSync(path.join(auditDir, "reports", "2026-10-01-website-audit.md"), REPORT, "utf-8");
    process.env.DASHBOARD_DATA_DIR = dataDir;
    process.env.WEBSITE_AUDIT_DIR = auditDir;
  });

  it("captures a multi-line body/recommendation, not just inline ones", async () => {
    const { importNewAuditReports, listAudits, getAuditDetail } = await import("@/lib/issues/auditImport");
    importNewAuditReports();
    const audits = listAudits();
    expect(audits).toHaveLength(1);
    const detail = getAuditDetail(audits[0]!.id);
    expect(detail!.findings).toHaveLength(2);

    const multiLine = detail!.findings.find((f) => f.title.includes("Multi-line finding"));
    expect(multiLine!.body).toContain("This description is on its own line below the title.");
    expect(multiLine!.recommendation).toBe("Fix the thing described above.");

    const inline = detail!.findings.find((f) => f.title.includes("Inline finding"));
    expect(inline!.body).toContain("This description stays on the same line as the title.");
    expect(inline!.recommendation).toBe("Fix this other thing too.");
  });
});
