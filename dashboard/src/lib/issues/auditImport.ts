import "server-only";

import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { getHealthDb } from "../db";
import { AUDIT_REPORTS_DIR } from "./config";
import { computeAuditScore } from "./scoring";

/**
 * Reads the existing monthly Markdown audit reports (never writes to them)
 * and indexes them into audit_imports / content_findings, syncing findings
 * into the shared issues table (site is always "retail" — the monthly
 * audit covers studiostonecreative.com only).
 */

const REPORT_FILENAME = /^(\d{4}-\d{2}-\d{2})-website-audit\.md$/;

interface ParsedFinding {
  priority: string | null;
  title: string;
  body: string;
}

function fingerprintFinding(title: string): string {
  const normalized = title
    .toLowerCase()
    .replace(/[^a-z0-9 ]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .slice(0, 8)
    .join(" ");
  return createHash("sha1").update(`audit|${normalized}`).digest("hex").slice(0, 16);
}

function guessCategory(title: string, body: string): string {
  const text = `${title} ${body}`.toLowerCase();
  if (/mobile|390px|viewport/.test(text)) return "ux";
  if (/accessib|alt text|aria|contrast|heading/.test(text)) return "accessibility";
  if (/meta description|seo|title tag|h1|canonical/.test(text)) return "content";
  if (/translat|locale|language|french|german|fr_fr|de_de/.test(text)) return "localization";
  if (/form|submit|field|captcha/.test(text)) return "forms";
  if (/nav|menu|header|footer|link/.test(text)) return "navigation";
  if (/load time|performance|slow|lighthouse|speed/.test(text)) return "performance";
  if (/journey|checkout|cart|booking|order form/.test(text)) return "journey";
  return "technical";
}

function extractSection(markdown: string, heading: string): string | null {
  const re = new RegExp(`##\\s+${heading}\\s*\\n([\\s\\S]*?)(?=\\n##\\s|$)`, "i");
  const match = markdown.match(re);
  return match ? (match[1] ?? "").trim() : null;
}

function parseFindings(markdown: string): ParsedFinding[] {
  const findings: ParsedFinding[] = [];
  const findingsSection = extractSection(markdown, "Findings") ?? markdown;
  const priorityBlocks = findingsSection.split(/(?=^###\s+P[0-3]\b)/m);

  for (const block of priorityBlocks) {
    const headingMatch = block.match(/^###\s+(P[0-3])\b/m);
    const blockPriority = headingMatch ? (headingMatch[1] ?? null) : null;
    // Body capture must not require a leading newline — compact P2/P3 bullets
    // put the description straight after the closing "**" on the same line,
    // while other entries put it on the following line(s). The lookahead's
    // end-of-input branch MUST be a true end-of-string assertion, not a bare
    // `$` — under the /m flag, `$` also matches the end of the title's own
    // line, so a lazy `[\s\S]*?` would stop there immediately (zero-length
    // body) for any entry whose description starts on a new line.
    const entryRe =
      /(?:^\*\*\d+\.\s+(.+?)\*\*|^####\s+P[0-3]-\d+\.\s+(.+?)$)([\s\S]*?)(?=\n(?:\*\*\d+\.|####\s+P[0-3]-\d+\.|###\s+P[0-3]\b)|$(?![\s\S]))/gm;
    let m: RegExpExecArray | null;
    while ((m = entryRe.exec(block)) !== null) {
      const title = (m[1] ?? m[2] ?? "").trim();
      const body = (m[3] ?? "").trim();
      if (title) findings.push({ priority: blockPriority, title, body });
    }
  }
  return findings;
}

function parseStatus(markdown: string): string {
  if (/AUDIT PARTIALLY COMPLETE/i.test(markdown)) return "AUDIT PARTIALLY COMPLETE";
  if (/AUDIT COMPLETE/i.test(markdown)) return "AUDIT COMPLETE";
  return "UNKNOWN";
}

function resolveAuditIssue(findingFingerprint: string): void {
  getHealthDb()
    .prepare(`UPDATE issues SET status = 'resolved' WHERE fingerprint = ? AND source = 'full_audit' AND status != 'resolved'`)
    .run(`audit:${findingFingerprint}`);
}

const PRIORITY_TO_SEVERITY: Record<string, string> = { P0: "critical", P1: "warning", P2: "warning", P3: "info" };

function syncIssuesFromFindings(findings: ParsedFinding[], auditDate: string): void {
  const db = getHealthDb();
  const upsertIssue = db.prepare(`
    INSERT INTO issues (fingerprint, severity, category, site, page, title, description, evidence, source, first_detected, last_detected, frequency)
    VALUES (@fingerprint, @severity, @category, 'retail', @page, @title, @description, @evidence, 'full_audit', @auditDate, @auditDate, 1)
    ON CONFLICT(fingerprint) DO UPDATE SET
      last_detected = @auditDate,
      frequency = frequency + 1,
      current_value = excluded.description,
      status = CASE WHEN issues.status = 'resolved' THEN 'open' ELSE issues.status END
  `);

  for (const f of findings) {
    const fp = `audit:${fingerprintFinding(f.title)}`;
    const pageMatch = f.body.match(/\*\*URL:\*\*\s*`?([^`\n]+)`?/i);
    const recMatch = f.body.match(/\*\*Recommendation:\*\*\s*([^\n]+)/i);
    upsertIssue.run({
      fingerprint: fp,
      severity: PRIORITY_TO_SEVERITY[f.priority ?? ""] ?? "info",
      category: guessCategory(f.title, f.body),
      page: pageMatch ? (pageMatch[1] ?? "").trim() : null,
      title: f.title,
      description: f.body.slice(0, 2000),
      evidence: recMatch ? (recMatch[1] ?? "").trim() : null,
      auditDate,
    });
  }
}

export interface ImportSummary {
  imported: number;
  skipped: number;
  files: string[];
}

export function importNewAuditReports(): ImportSummary {
  const db = getHealthDb();
  let files: string[] = [];
  try {
    files = readdirSync(AUDIT_REPORTS_DIR).filter((f) => REPORT_FILENAME.test(f));
  } catch {
    return { imported: 0, skipped: 0, files: [] };
  }

  const existing = new Set(
    (db.prepare("SELECT report_path FROM audit_imports").all() as { report_path: string }[]).map((r) => r.report_path),
  );

  let imported = 0;
  let skipped = 0;
  const importedFiles: string[] = [];
  const sorted = [...files].sort();

  for (const file of sorted) {
    const fullPath = path.join(AUDIT_REPORTS_DIR, file);
    if (existing.has(fullPath)) {
      skipped++;
      continue;
    }
    const match = file.match(REPORT_FILENAME);
    if (!match) {
      skipped++;
      continue;
    }
    const auditDate = match[1] ?? "";
    const markdown = readFileSync(fullPath, "utf-8");
    const status = parseStatus(markdown);
    const executiveSummary = extractSection(markdown, "Executive Summary");
    const findings = parseFindings(markdown);

    const critical = findings.filter((f) => f.priority === "P0").length;
    const warning = findings.filter((f) => f.priority === "P1" || f.priority === "P2").length;
    const score = computeAuditScore(findings.map((f) => ({ priority: f.priority })));

    const result = db
      .prepare(
        `INSERT INTO audit_imports (report_path, audit_date, status, executive_summary, critical_count, warning_count, health_score) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(fullPath, auditDate, status, executiveSummary, critical, warning, score);
    const auditImportId = result.lastInsertRowid as number;

    const previous = db.prepare(`SELECT id FROM audit_imports WHERE audit_date < ? ORDER BY audit_date DESC LIMIT 1`).get(auditDate) as
      | { id: number }
      | undefined;
    const previousFingerprints = previous
      ? new Set(
          (db.prepare("SELECT fingerprint FROM content_findings WHERE audit_import_id = ?").all(previous.id) as { fingerprint: string }[]).map(
            (r) => r.fingerprint,
          ),
        )
      : new Set<string>();

    const insertFinding = db.prepare(
      `INSERT INTO content_findings (audit_import_id, fingerprint, priority, category, page, title, body, recommendation, change_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    );
    const currentFingerprints = new Set<string>();
    for (const f of findings) {
      const fp = fingerprintFinding(f.title);
      currentFingerprints.add(fp);
      const pageMatch = f.body.match(/\*\*URL:\*\*\s*`?([^`\n]+)`?/i);
      const recMatch = f.body.match(/\*\*Recommendation:\*\*\s*([^\n]+)/i);
      insertFinding.run(
        auditImportId,
        fp,
        f.priority,
        guessCategory(f.title, f.body),
        pageMatch ? (pageMatch[1] ?? "").trim() : null,
        f.title,
        f.body.slice(0, 4000),
        recMatch ? (recMatch[1] ?? "").trim() : null,
        previousFingerprints.has(fp) ? "ongoing" : "new",
      );
    }

    if (previous) {
      const vanished = [...previousFingerprints].filter((fp) => !currentFingerprints.has(fp));
      if (vanished.length > 0) {
        const markResolved = db.prepare(`UPDATE content_findings SET change_status = 'resolved' WHERE audit_import_id = ? AND fingerprint = ?`);
        for (const fp of vanished) markResolved.run(previous.id, fp);
        for (const fp of vanished) resolveAuditIssue(fp);
      }
    }

    syncIssuesFromFindings(findings, auditDate);
    imported++;
    importedFiles.push(file);
  }

  return { imported, skipped, files: importedFiles };
}

export interface LatestAudit {
  audit_date: string;
  health_score: number | null;
  status: string | null;
}

export function getLatestAudit(): LatestAudit | null {
  importNewAuditReports();
  return (
    (getHealthDb().prepare(`SELECT audit_date, health_score, status FROM audit_imports ORDER BY audit_date DESC LIMIT 1`).get() as
      | LatestAudit
      | undefined) ?? null
  );
}

export interface AuditImportRow {
  id: number;
  report_path: string;
  audit_date: string;
  status: string | null;
  executive_summary: string | null;
  critical_count: number;
  warning_count: number;
  health_score: number | null;
  imported_at: string;
}

/** All imported monthly audits, most recent first. Runs the same import-new-reports
 * sweep as getLatestAudit so a page load always reflects any report dropped on disk. */
export function listAudits(): AuditImportRow[] {
  importNewAuditReports();
  return getHealthDb().prepare(`SELECT * FROM audit_imports ORDER BY audit_date DESC`).all() as AuditImportRow[];
}

export interface ContentFindingRow {
  id: number;
  fingerprint: string;
  priority: string | null;
  category: string | null;
  page: string | null;
  title: string;
  body: string | null;
  recommendation: string | null;
  change_status: string | null;
}

export interface AuditDetail extends AuditImportRow {
  findings: ContentFindingRow[];
  /** Raw Markdown of the full report, read fresh from disk each call (not
   * cached) so this always reflects the current file content. Null if the
   * file has since moved or been deleted. */
  reportMarkdown: string | null;
}

export function getAuditDetail(id: number): AuditDetail | null {
  const db = getHealthDb();
  const audit = db.prepare(`SELECT * FROM audit_imports WHERE id = ?`).get(id) as AuditImportRow | undefined;
  if (!audit) return null;
  const findings = db
    .prepare(`SELECT * FROM content_findings WHERE audit_import_id = ? ORDER BY priority ASC, id ASC`)
    .all(id) as ContentFindingRow[];
  let reportMarkdown: string | null = null;
  try {
    reportMarkdown = readFileSync(audit.report_path, "utf-8");
  } catch {
    reportMarkdown = null;
  }
  return { ...audit, findings, reportMarkdown };
}
