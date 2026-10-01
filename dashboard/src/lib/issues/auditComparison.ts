// dashboard/src/lib/issues/auditComparison.ts
import "server-only";

import { getHealthDb } from "../db";
import { importNewAuditReports, type AuditImportRow, type ContentFindingRow } from "./auditImport";

/**
 * Pure read of data auditImport.ts already computed and stored — no new
 * scoring or fingerprinting logic here. "Fixed" is read directly off the
 * PREVIOUS audit's findings (marked change_status='resolved' when their
 * fingerprint vanished from the current audit); "new"/"ongoing" are read
 * off the CURRENT audit's findings, exactly as auditImport.ts set them at
 * import time.
 */

const PRIORITY_ORDER: Record<string, number> = { P0: 0, P1: 1, P2: 2, P3: 3 };

export interface AuditComparison {
  current: AuditImportRow;
  previous: AuditImportRow | null;
  scoreDelta: number | null;
  fixed: ContentFindingRow[];
  newFindings: ContentFindingRow[];
  stillOpen: ContentFindingRow[];
  /** Currently-open findings (new + still-open) with a recommendation, ranked
   * P0 first, capped to a manageable "what to do next" list. */
  furtherImprovements: ContentFindingRow[];
}

/** Comparison between the two most recent imported audits. Runs the same
 * import-new-reports sweep as the rest of auditImport.ts so this reflects
 * any report just dropped on disk. Returns null only if no audit exists yet. */
export function compareLatestAudits(): AuditComparison | null {
  importNewAuditReports();
  const db = getHealthDb();
  const audits = db.prepare(`SELECT * FROM audit_imports ORDER BY audit_date DESC LIMIT 2`).all() as AuditImportRow[];
  const current = audits[0];
  if (!current) return null;
  const previous = audits[1] ?? null;

  const currentFindings = db
    .prepare(`SELECT * FROM content_findings WHERE audit_import_id = ? ORDER BY priority ASC, id ASC`)
    .all(current.id) as ContentFindingRow[];

  const fixed = previous
    ? (db
        .prepare(`SELECT * FROM content_findings WHERE audit_import_id = ? AND change_status = 'resolved' ORDER BY priority ASC, id ASC`)
        .all(previous.id) as ContentFindingRow[])
    : [];

  const newFindings = currentFindings.filter((f) => f.change_status === "new");
  const stillOpen = currentFindings.filter((f) => f.change_status === "ongoing");

  const furtherImprovements = [...newFindings, ...stillOpen]
    .filter((f) => f.recommendation)
    .sort((a, b) => (PRIORITY_ORDER[a.priority ?? ""] ?? 9) - (PRIORITY_ORDER[b.priority ?? ""] ?? 9))
    .slice(0, 8);

  const scoreDelta =
    previous && current.health_score != null && previous.health_score != null ? current.health_score - previous.health_score : null;

  return { current, previous, scoreDelta, fixed, newFindings, stillOpen, furtherImprovements };
}
