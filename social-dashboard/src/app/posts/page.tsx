"use client";

import { useCallback, useEffect, useState } from "react";
import { Page } from "@/components/ui/Page";
import { StatCard } from "@/components/ui/StatCard";
import { Panel } from "@/components/ui/Panel";
import { DataTable } from "@/components/ui/DataTable";
import { formatNumber, formatDate, formatCurrency } from "@/lib/format";

type Platform = "instagram" | "facebook" | "tiktok" | "pinterest";
type PlatformFilter = "all" | Platform;

interface SocialPost {
  id: number;
  posted_date: string;
  platform: Platform;
  post_type: string | null;
  caption: string | null;
  likes: number;
  comments: number;
  shares: number;
  link_clicks: number;
  reach: number | null;
  views: number | null;
  utm_campaign: string | null;
  revenue_attributed: number;
  /** The planner post this entry belongs to, if any. */
  planner_idea_id: number | null;
}

interface PlatformInsightRow {
  platform: Platform;
  metric: string;
  period_start: string;
  period_end: string;
  value: number | null;
  unit: string | null;
  change_vs_prev_period_pct: number | null;
}

const PLATFORMS: Platform[] = ["instagram", "facebook", "tiktok", "pinterest"];

const EMPTY_FORM = {
  posted_date: "",
  platform: "instagram" as Platform,
  post_type: "",
  caption: "",
  reach: "",
  likes: "",
  comments: "",
  shares: "",
  link_clicks: "",
  utm_source: "",
  utm_medium: "",
  utm_campaign: "",
  revenue_attributed: "",
};

const PLATFORM_LABEL: Record<Platform, string> = { instagram: "Instagram", facebook: "Facebook", tiktok: "TikTok", pinterest: "Pinterest" };

type FormKey = keyof typeof EMPTY_FORM;
const FORM_FIELDS: { key: FormKey; label: string; type?: string; placeholder?: string; wide?: boolean }[] = [
  { key: "posted_date", label: "Posted", type: "date" },
  { key: "post_type", label: "Type", placeholder: "reel, carousel…" },
  { key: "caption", label: "Caption", wide: true },
  { key: "reach", label: "Reach", type: "number" },
  { key: "likes", label: "Likes", type: "number" },
  { key: "comments", label: "Comments", type: "number" },
  { key: "shares", label: "Shares", type: "number" },
  { key: "link_clicks", label: "Link clicks", type: "number" },
  { key: "revenue_attributed", label: "Revenue", type: "number" },
  { key: "utm_campaign", label: "UTM campaign", placeholder: "fall-launch" },
  { key: "utm_source", label: "UTM source", placeholder: "instagram" },
  { key: "utm_medium", label: "UTM medium", placeholder: "organic-social" },
];


export default function SocialMediaPage() {
  const [filter, setFilter] = useState<PlatformFilter>("all");
  const [posts, setPosts] = useState<SocialPost[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [insights, setInsights] = useState<PlatformInsightRow[]>([]);
  const [insightsLoading, setInsightsLoading] = useState(true);
  const [importing, setImporting] = useState(false);
  const [importMessage, setImportMessage] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);

  const load = useCallback(async (platform: PlatformFilter) => {
    const qs = platform === "all" ? "" : `?platform=${platform}`;
    const res = await fetch(`/api/marketing/social${qs}`, { cache: "no-store" });
    if (res.ok) setPosts((await res.json()).posts);
    setLoading(false);
  }, []);

  const loadInsights = useCallback(async () => {
    const res = await fetch("/api/marketing/social/insights", { cache: "no-store" });
    if (res.ok) setInsights((await res.json()).insights);
    setInsightsLoading(false);
  }, []);

  useEffect(() => {
    setLoading(true);
    load(filter);
  }, [filter, load]);

  useEffect(() => {
    loadInsights();
  }, [loadInsights]);

  async function handleImportFile(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    if (files.length === 0) return;
    setImporting(true);
    setImportMessage(null);
    const totals = { added: 0, updated: 0, skipped: 0, summaries: 0 };
    const failures: string[] = [];
    try {
      // Facebook and Instagram come as separate files, so take several at once.
      for (const file of files) {
        const res = await fetch("/api/marketing/social/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ csv: await file.text() }),
        }).catch(() => null);
        if (!res?.ok) {
          const body = res ? await res.json().catch(() => ({})) : {};
          failures.push(`${file.name}: ${body.error ?? "couldn't be imported"}`);
          continue;
        }
        const { result } = await res.json();
        totals.added += result.postsImported;
        totals.updated += result.postsUpdated ?? 0;
        totals.skipped += result.postsSkipped;
        totals.summaries += result.summariesImported;
      }
      const parts = [
        `${totals.added} new post${totals.added === 1 ? "" : "s"} added`,
        totals.updated > 0 ? `${totals.updated} updated with the latest numbers` : null,
        totals.skipped > 0 ? `${totals.skipped} already imported` : null,
        totals.summaries > 0 ? `${totals.summaries} period totals refreshed` : null,
      ].filter(Boolean);
      setImportMessage([parts.join(", ") + ".", ...failures].join(" "));
      load(filter);
      loadInsights();
    } finally {
      setImporting(false);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    const res = await fetch("/api/marketing/social", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        posted_date: form.posted_date,
        platform: form.platform,
        post_type: form.post_type || null,
        caption: form.caption || null,
        reach: form.reach ? Number(form.reach) : null,
        likes: Number(form.likes) || 0,
        comments: Number(form.comments) || 0,
        shares: Number(form.shares) || 0,
        link_clicks: Number(form.link_clicks) || 0,
        utm_source: form.utm_source || null,
        utm_medium: form.utm_medium || null,
        utm_campaign: form.utm_campaign || null,
        revenue_attributed: Number(form.revenue_attributed) || 0,
      }),
    });
    setSaving(false);
    if (res.ok) {
      setForm(EMPTY_FORM);
      setShowForm(false);
      load(filter);
    }
  }

  async function handleDelete(id: number) {
    await fetch(`/api/marketing/social/${id}`, { method: "DELETE" });
    setConfirmDelete(null);
    load(filter);
  }

  const totalEngagement = posts.reduce((sum, p) => sum + p.likes + p.comments + p.shares, 0);
  const totalClicks = posts.reduce((sum, p) => sum + p.link_clicks, 0);

  const insightsByPlatform = new Map<Platform, PlatformInsightRow[]>();
  for (const row of insights) {
    const list = insightsByPlatform.get(row.platform) ?? [];
    list.push(row);
    insightsByPlatform.set(row.platform, list);
  }

  return (
    <Page
      title="Post log"
      description="Everything that's gone out, with its numbers. Meta exports fill this in, and posts marked as posted in the planner appear here too. Everything here also shows on the calendar."
    >
      <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 14 }}>
        <StatCard label={filter === "all" ? "Posts" : `${PLATFORM_LABEL[filter]} posts`} value={String(posts.length)} />
        <StatCard label="Engagement" value={formatNumber(totalEngagement)} />
        <StatCard label="Link clicks" value={formatNumber(totalClicks)} />
      </section>

      <Panel
        title="Posts"
        headerAction={
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <div role="tablist" aria-label="Filter by platform" style={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
              {(["all", ...PLATFORMS] as PlatformFilter[]).map((p) => {
                const active = p === filter;
                return (
                  <button
                    key={p}
                    role="tab"
                    aria-selected={active}
                    type="button"
                    onClick={() => setFilter(p)}
                    className="btn btn-sm"
                    style={{ borderColor: active ? "var(--border-strong)" : "transparent", background: active ? "var(--surface-raised)" : "transparent", color: active ? "var(--text)" : "var(--text-soft)" }}
                  >
                    {p === "all" ? "All" : PLATFORM_LABEL[p]}
                  </button>
                );
              })}
            </div>
            <button type="button" className={`btn btn-sm ${showForm ? "btn-ghost" : ""}`} onClick={() => setShowForm((v) => !v)}>
              {showForm ? "Cancel" : "Log a post"}
            </button>
          </div>
        }
      >
        {showForm && (
          <form onSubmit={handleSubmit} className="card-quiet" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 14, marginBottom: 20, padding: 16 }}>
            <div className="field">
              <label htmlFor="post-platform">Platform</label>
              <select id="post-platform" className="input" value={form.platform} onChange={(e) => setForm({ ...form, platform: e.target.value as Platform })}>
                {PLATFORMS.map((p) => (
                  <option key={p} value={p}>
                    {PLATFORM_LABEL[p]}
                  </option>
                ))}
              </select>
            </div>
            {FORM_FIELDS.map((f) => (
              <div key={f.key} className="field" style={f.wide ? { gridColumn: "span 2" } : undefined}>
                <label htmlFor={`post-${f.key}`}>{f.label}</label>
                <input
                  id={`post-${f.key}`}
                  className="input"
                  type={f.type ?? "text"}
                  required={f.key === "posted_date"}
                  min={f.type === "number" ? 0 : undefined}
                  step={f.key === "revenue_attributed" ? "0.01" : undefined}
                  placeholder={f.placeholder}
                  value={form[f.key]}
                  onChange={(e) => setForm({ ...form, [f.key]: e.target.value })}
                />
              </div>
            ))}
            <div style={{ gridColumn: "1 / -1" }}>
              <button type="submit" className="btn btn-primary" disabled={saving}>
                {saving ? "Saving…" : "Save post"}
              </button>
            </div>
          </form>
        )}

        {loading ? (
          <p className="muted">Loading…</p>
        ) : (
          <DataTable
            emptyText="No posts logged yet."
            rows={posts}
            columns={[
              { header: "Date", render: (p) => formatDate(p.posted_date) },
              { header: "Platform", render: (p) => PLATFORM_LABEL[p.platform] ?? p.platform },
              { header: "Type", render: (p) => <span className="muted">{p.post_type ?? "—"}</span> },
              {
                header: "Planner",
                render: (p) =>
                  p.planner_idea_id != null ? (
                    <a href={`/?idea=${p.planner_idea_id}`} className="pill pill-second" style={{ textDecoration: "none" }} title="Open this post in the planner">
                      Open in planner
                    </a>
                  ) : (
                    <span style={{ color: "var(--text-faint)" }}>—</span>
                  ),
              },
              { header: "Views", render: (p) => (p.views == null ? "—" : formatNumber(p.views)), align: "right" },
              { header: "Reach", render: (p) => (p.reach === null ? "—" : formatNumber(p.reach)), align: "right" },
              { header: "Likes", render: (p) => formatNumber(p.likes), align: "right" },
              { header: "Comments", render: (p) => formatNumber(p.comments), align: "right" },
              { header: "Shares", render: (p) => formatNumber(p.shares), align: "right" },
              { header: "Clicks", render: (p) => formatNumber(p.link_clicks), align: "right" },
              { header: "Campaign", render: (p) => <span className="muted">{p.utm_campaign ?? "—"}</span> },
              { header: "Revenue", render: (p) => (p.revenue_attributed > 0 ? formatCurrency(p.revenue_attributed) : "—"), align: "right" },
              {
                header: "",
                render: (p) =>
                  confirmDelete === p.id ? (
                    <span style={{ display: "inline-flex", gap: 10 }}>
                      <button type="button" className="link-btn danger" style={{ color: "var(--negative)" }} onClick={() => handleDelete(p.id)}>
                        Delete
                      </button>
                      <button type="button" className="link-btn" onClick={() => setConfirmDelete(null)}>
                        Keep
                      </button>
                    </span>
                  ) : (
                    <button type="button" className="link-btn danger" onClick={() => setConfirmDelete(p.id)}>
                      Delete
                    </button>
                  ),
                align: "right",
              },
            ]}
          />
        )}
      </Panel>

      <Panel
        title="Import from Meta"
        subtitle="Upload Meta Business Suite exports: the Facebook and Instagram post files together, or a summary export"
        about={
          <p>
            Re-uploading the same or an overlapping export is safe. Posts already in the log get the latest numbers instead of a second row, and
            period totals are updated in place.
          </p>
        }
        headerAction={
          <label className={`btn btn-sm ${importing ? "" : "btn-primary"}`} style={{ opacity: importing ? 0.6 : 1, cursor: importing ? "default" : "pointer" }}>
            {importing ? "Importing…" : "Choose CSV files"}
            <input type="file" accept=".csv,text/csv" multiple onChange={handleImportFile} disabled={importing} style={{ display: "none" }} />
          </label>
        }
      >
        {importMessage ? <p style={{ fontSize: 13, margin: 0 }}>{importMessage}</p> : <p className="muted" style={{ fontSize: 13, margin: 0 }}>No file chosen.</p>}
      </Panel>

      <Panel title="Platform totals" subtitle="Period totals from the latest Meta export">
        {insightsLoading ? (
          <p className="muted">Loading…</p>
        ) : insights.length === 0 ? (
          <p className="muted" style={{ fontSize: 13 }}>
            Nothing imported yet. Upload a CSV export above.
          </p>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 340px), 1fr))", gap: 24 }}>
            {Array.from(insightsByPlatform.entries()).map(([platform, rows]) => (
              <div key={platform}>
                <p style={{ fontSize: 13, fontWeight: 600, margin: "0 0 2px" }}>{PLATFORM_LABEL[platform] ?? platform}</p>
                <p style={{ fontSize: 12, color: "var(--text-faint)", margin: "0 0 10px" }}>
                  {formatDate(rows[0]!.period_start)} to {formatDate(rows[0]!.period_end)}
                </p>
                <DataTable
                  emptyText="No metrics."
                  rows={rows}
                  columns={[
                    { header: "Metric", render: (r) => r.metric },
                    { header: "Value", render: (r) => (r.value === null ? "—" : r.unit === "percent" ? `${r.value.toFixed(1)}%` : formatNumber(r.value)), align: "right" },
                    {
                      header: "vs previous",
                      render: (r) =>
                        r.change_vs_prev_period_pct === null ? (
                          "—"
                        ) : (
                          <span style={{ color: r.change_vs_prev_period_pct >= 0 ? "var(--second)" : "var(--negative)" }}>
                            {r.change_vs_prev_period_pct >= 0 ? "+" : ""}
                            {r.change_vs_prev_period_pct.toFixed(1)}%
                          </span>
                        ),
                      align: "right",
                    },
                  ]}
                />
              </div>
            ))}
          </div>
        )}
      </Panel>
    </Page>
  );
}
