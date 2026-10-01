// dashboard/src/app/tools/audit-history/page.tsx
"use client";

import { useCallback, useEffect, useState } from "react";
import { Page } from "@/components/ui/Page";
import { Panel } from "@/components/ui/Panel";
import { DataTable } from "@/components/ui/DataTable";
import { Badge } from "@/components/ui/Badge";

interface AuditImportRow {
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

interface ContentFindingRow {
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

interface AuditDetail extends AuditImportRow {
  findings: ContentFindingRow[];
  reportMarkdown: string | null;
}

interface AuditComparison {
  current: AuditImportRow;
  previous: AuditImportRow | null;
  scoreDelta: number | null;
  fixed: ContentFindingRow[];
  newFindings: ContentFindingRow[];
  stillOpen: ContentFindingRow[];
  furtherImprovements: ContentFindingRow[];
}

/** The "Comparison to Previous Audit" content only — rendered inside the
 * expanded per-audit detail, below its Findings table, and only once that
 * audit's title has been clicked open (see AuditHistoryPage). */
function ComparisonSummary({ comparison }: { comparison: AuditComparison }) {
  const { current, previous, scoreDelta, fixed, newFindings, stillOpen } = comparison;

  if (!previous) {
    return (
      <p style={{ fontSize: 13, color: "var(--text-soft)" }}>
        This is the first imported audit ({new Date(current.audit_date).toLocaleDateString()}) — there&apos;s nothing yet to compare it
        against. Comparison will appear here once a second audit is imported.
      </p>
    );
  }

  const deltaColor = scoreDelta == null ? undefined : scoreDelta > 0 ? "var(--positive)" : scoreDelta < 0 ? "var(--negative)" : "var(--text-soft)";
  const deltaLabel = scoreDelta == null ? "—" : scoreDelta > 0 ? `+${scoreDelta}` : `${scoreDelta}`;

  return (
    <div>
      <p style={{ fontSize: 12, color: "var(--text-soft)", marginBottom: 14 }}>
        {new Date(previous.audit_date).toLocaleDateString()} ({previous.health_score ?? "—"}) → {new Date(current.audit_date).toLocaleDateString()} ({current.health_score ?? "—"})
        , computed from each finding&apos;s tracked status across the two audits — not re-derived or guessed.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12 }}>
        <div className="bracket-panel" style={{ padding: 14, textAlign: "center" }}>
          <div style={{ fontSize: 11, color: "var(--text-soft)", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 6 }}>Score Change</div>
          <div style={{ fontSize: 24, fontWeight: 600, color: deltaColor }}>{deltaLabel}</div>
        </div>
        <div className="bracket-panel" style={{ padding: 14, textAlign: "center" }}>
          <div style={{ fontSize: 11, color: "var(--text-soft)", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 6 }}>Fixed</div>
          <div style={{ fontSize: 24, fontWeight: 600, color: "var(--positive)" }}>{fixed.length}</div>
        </div>
        <div className="bracket-panel" style={{ padding: 14, textAlign: "center" }}>
          <div style={{ fontSize: 11, color: "var(--text-soft)", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 6 }}>New</div>
          <div style={{ fontSize: 24, fontWeight: 600, color: "var(--warning)" }}>{newFindings.length}</div>
        </div>
        <div className="bracket-panel" style={{ padding: 14, textAlign: "center" }}>
          <div style={{ fontSize: 11, color: "var(--text-soft)", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 6 }}>Still Open</div>
          <div style={{ fontSize: 24, fontWeight: 600 }}>{stillOpen.length}</div>
        </div>
      </div>

      {fixed.length > 0 && (
        <div style={{ marginTop: 18 }}>
          <strong style={{ fontSize: 12.5 }}>Fixed since {new Date(previous.audit_date).toLocaleDateString()}</strong>
          <ul style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 12.5, color: "var(--text-soft)", lineHeight: 1.7 }}>
            {fixed.map((f) => (
              <li key={f.id}>{f.title}</li>
            ))}
          </ul>
        </div>
      )}

      {newFindings.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <strong style={{ fontSize: 12.5 }}>New this audit</strong>
          <ul style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 12.5, color: "var(--text-soft)", lineHeight: 1.7 }}>
            {newFindings.map((f) => (
              <li key={f.id}>
                {f.priority && <Badge variant="severity">{f.priority}</Badge>} {f.title}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function FurtherImprovementsPanel({ comparison }: { comparison: AuditComparison }) {
  const { furtherImprovements } = comparison;
  return (
    <Panel title="Further Improvements">
      <p style={{ fontSize: 12, color: "var(--text-soft)", marginBottom: 14 }}>
        Every currently-open finding&apos;s own recommendation (new + still-open), ranked by priority — not a separate suggestion
        generator, just the highest-leverage next steps already on record.
      </p>
      {furtherImprovements.length === 0 ? (
        <p style={{ fontSize: 13, color: "var(--text-soft)" }}>No open findings with a recorded recommendation.</p>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {furtherImprovements.map((f) => (
            <div key={f.id} className="bracket-panel" style={{ padding: 12, display: "flex", gap: 12, alignItems: "flex-start" }}>
              {f.priority && (
                <div style={{ flexShrink: 0 }}>
                  <Badge variant="severity">{f.priority}</Badge>
                </div>
              )}
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 2 }}>{f.title}</div>
                <div style={{ fontSize: 12.5, color: "var(--text-soft)" }}>{f.recommendation}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

export default function AuditHistoryPage() {
  const [audits, setAudits] = useState<AuditImportRow[]>([]);
  const [comparison, setComparison] = useState<AuditComparison | null>(null);
  const [selected, setSelected] = useState<AuditDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [fullReportOpen, setFullReportOpen] = useState(false);

  useEffect(() => {
    fetch("/api/audits", { cache: "no-store" })
      .then((res) => res.json())
      .then((body) => setAudits(body.audits ?? []))
      .finally(() => setLoading(false));
    fetch("/api/audits/comparison", { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : null))
      .then((body) => setComparison(body?.comparison ?? null));
  }, []);

  const openAudit = useCallback(async (id: number) => {
    setFullReportOpen(false);
    setDetailLoading(true);
    const res = await fetch(`/api/audits/${id}`, { cache: "no-store" });
    if (res.ok) {
      const body = await res.json();
      setSelected(body.audit);
    }
    setDetailLoading(false);
  }, []);

  return (
    <Page title="Audit History" description="Imported website-audit reports, their executive summaries, and per-finding detail.">
      <Panel title="Audit Reports">
        {loading ? (
          <p style={{ color: "var(--text-soft)", fontSize: 13 }}>Loading…</p>
        ) : (
          <DataTable
            emptyText="No audit reports have been imported yet."
            rows={audits}
            onRowClick={(row) => openAudit(row.id)}
            columns={[
              { header: "Date", render: (a) => new Date(a.audit_date).toLocaleDateString() },
              { header: "Status", render: (a) => a.status ?? "—" },
              { header: "Health Score", render: (a) => (a.health_score != null ? a.health_score.toFixed(0) : "—") },
              { header: "Critical", render: (a) => <span style={{ color: a.critical_count > 0 ? "var(--negative)" : undefined }}>{a.critical_count}</span> },
              { header: "Warnings", render: (a) => <span style={{ color: a.warning_count > 0 ? "var(--warning)" : undefined }}>{a.warning_count}</span> },
              { header: "Summary", render: (a) => <span style={{ color: "var(--text-soft)" }}>{a.executive_summary?.slice(0, 90) ?? "—"}</span> },
            ]}
          />
        )}
      </Panel>

      {comparison && <FurtherImprovementsPanel comparison={comparison} />}

      {(detailLoading || selected) && (
        <Panel
          title={selected ? `Audit Report — ${new Date(selected.audit_date).toLocaleDateString()}` : "Loading…"}
          headerAction={
            selected && (
              <button
                className="font-mono"
                onClick={() => setSelected(null)}
                style={{ fontSize: 11, background: "none", border: "1px solid var(--border)", borderRadius: 6, padding: "6px 12px", color: "var(--text-soft)", cursor: "pointer" }}
              >
                Close
              </button>
            )
          }
        >
          {detailLoading ? (
            <p style={{ color: "var(--text-soft)", fontSize: 13 }}>Loading…</p>
          ) : (
            <>
              {selected?.executive_summary && <p style={{ fontSize: 13, color: "var(--text-soft)", marginBottom: 16 }}>{selected.executive_summary}</p>}

              <strong style={{ fontSize: 12.5, display: "block", marginBottom: 8 }}>Findings</strong>
              <DataTable
                emptyText="No findings recorded for this audit."
                rows={selected?.findings ?? []}
                columns={[
                  { header: "Priority", render: (f) => (f.priority ? <Badge variant="severity">{f.priority}</Badge> : "—") },
                  { header: "Category", render: (f) => f.category ?? "—" },
                  { header: "Page", render: (f) => f.page ?? "—" },
                  { header: "Finding", render: (f) => f.title },
                  { header: "Recommendation", render: (f) => <span style={{ color: "var(--text-soft)" }}>{f.recommendation ?? "—"}</span> },
                  { header: "Status", render: (f) => f.change_status ?? "—" },
                ]}
              />

              {/* Comparison only exists for the latest audit (it's computed against the one before it) — hidden entirely for older rows. */}
              {selected && comparison && comparison.current.id === selected.id && (
                <div style={{ marginTop: 24, paddingTop: 20, borderTop: "1px solid var(--border)" }}>
                  <strong style={{ fontSize: 12.5, display: "block", marginBottom: 12 }}>Comparison to Previous Audit</strong>
                  <ComparisonSummary comparison={comparison} />
                </div>
              )}

              {selected && (
                <div style={{ marginTop: 24, paddingTop: 20, borderTop: "1px solid var(--border)" }}>
                  <button
                    className="font-mono"
                    onClick={() => setFullReportOpen((v) => !v)}
                    style={{ fontSize: 11, background: "none", border: "1px solid var(--border)", borderRadius: 6, padding: "6px 12px", color: "var(--text)", cursor: "pointer", textTransform: "uppercase", letterSpacing: 0.5 }}
                  >
                    {fullReportOpen ? "Hide Full Report" : "Show Full Report"}
                  </button>
                  {fullReportOpen && (
                    <pre
                      style={{
                        marginTop: 14,
                        padding: 16,
                        background: "var(--surface-alt)",
                        borderRadius: 8,
                        fontSize: 12,
                        lineHeight: 1.6,
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                        maxHeight: 600,
                        overflowY: "auto",
                      }}
                    >
                      {selected.reportMarkdown ?? "Report file is no longer available on disk."}
                    </pre>
                  )}
                </div>
              )}
            </>
          )}
        </Panel>
      )}
    </Page>
  );
}
