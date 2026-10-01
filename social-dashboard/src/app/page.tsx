"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Page } from "@/components/ui/Page";
import { Panel } from "@/components/ui/Panel";
import { DataTable } from "@/components/ui/DataTable";
import { Badge } from "@/components/ui/Badge";
import { Planner } from "@/components/planner/Planner";
import type { Me, PlannerIdea } from "@/components/planner/types";
import type { CalendarCustomItem, CalendarOccasion, CalendarTradeShow } from "@/lib/social/calendarEntries";
import { formatNumber, formatCurrency, formatDate } from "@/lib/format";
import { scoreSocialPosts } from "@/lib/social/scoring";
import { buildUtmUrl, UTM_SOURCES, UTM_MEDIUMS, InvalidUtmUrlError, type UtmSource, type UtmMedium } from "@/lib/social/utm";
import { scoreTrendFit, hedgeForClassification, type TrendFactors } from "@/lib/social/trendScore";
import { TREND_PROVIDERS } from "@/lib/social/trendProviders";
import { evaluateLearningLoopOutcome } from "@/lib/social/learningLoopOutcome";

type Platform = "instagram" | "facebook" | "tiktok" | "pinterest";

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
  utm_source: string | null;
  utm_medium: string | null;
  revenue_attributed: number;
  planner_idea_id: number | null;
}

interface CampaignAttributionRow {
  utmCampaign: string;
  utmSource: string;
  utmMedium: string;
  sessions: number;
  conversionRate: number;
  orders: number;
  sales: number;
  averageOrderValue: number;
}

interface AttributionResult {
  configured: boolean;
  rows: CampaignAttributionRow[];
}

interface TrendObservation {
  id: number;
  term: string;
  platform: string | null;
  observed_date: string;
  product_relevance: number;
  commercial_intent: number;
  regional_momentum: number;
  historical_performance: number;
  seasonality_timing: number;
  content_suitability: number;
  inventory_availability: number;
  note: string | null;
  source_url: string | null;
}

const FACTOR_FIELDS: { key: keyof TrendFactors; label: string; bodyKey: string }[] = [
  { key: "productRelevance", label: "Product relevance", bodyKey: "product_relevance" },
  { key: "commercialIntent", label: "Commercial intent", bodyKey: "commercial_intent" },
  { key: "regionalMomentum", label: "Regional momentum", bodyKey: "regional_momentum" },
  { key: "historicalPerformance", label: "Historical performance", bodyKey: "historical_performance" },
  { key: "seasonalityTiming", label: "Seasonality / timing", bodyKey: "seasonality_timing" },
  { key: "contentSuitability", label: "Content suitability", bodyKey: "content_suitability" },
  { key: "inventoryAvailability", label: "Inventory availability", bodyKey: "inventory_availability" },
];

interface TrendRecommendation {
  id: number;
  trend_observation_id: number | null;
  term: string;
  score_at_recommendation: number;
  classification_at_recommendation: string;
  recommended_date: string;
  linked_social_post_id: number | null;
}

type InsightConfidence = "high" | "promising" | "experimental" | "insufficient";
type InsightKind = "fact" | "inference" | "recommendation";

interface Insight {
  id: string;
  kind: InsightKind;
  confidence: InsightConfidence;
  text: string;
}

const INSIGHT_KIND_LABEL: Record<InsightKind, string> = { fact: "Fact", inference: "Inference", recommendation: "Recommendation" };
const CONFIDENCE_LABEL: Record<InsightConfidence, string> = {
  high: "High confidence",
  promising: "Promising test",
  experimental: "Experimental",
  insufficient: "Insufficient data",
};

function AutoInsightsPanel() {
  const [insights, setInsights] = useState<Insight[]>([]);
  const [postCount, setPostCount] = useState(0);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    const res = await fetch("/api/social-intelligence/auto-insights", { cache: "no-store" });
    if (res.ok) {
      const body = await res.json();
      setInsights(body.insights);
      setPostCount(body.postCount);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <Panel
      title="What the numbers say"
      subtitle="Generated from imported Meta insights and Shopify UTM sales"
      about={
        <>
          <p>
            Built automatically from data that&apos;s already imported, so there&apos;s nothing to log by hand. Revenue and orders always
            outrank engagement, and every item is labelled fact, inference or recommendation with a confidence level.
          </p>
          <p>Live trend providers aren&apos;t connected yet, so this looks back at what was posted. Use Trends for forward-looking research.</p>
        </>
      }
      headerAction={
        <button type="button" className="btn btn-sm" onClick={load} disabled={loading}>
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      }
    >
      {loading && insights.length === 0 ? (
        <p className="muted">Loading…</p>
      ) : insights.length === 0 ? (
        <p className="muted" style={{ fontSize: 13 }}>
          {postCount === 0 ? "Nothing imported yet. Upload a Meta Business Suite export on the Post log page." : "No notable patterns yet."}
        </p>
      ) : (
        <div style={{ display: "flex", flexDirection: "column" }}>
          {insights.map((insight) => (
            <div key={insight.id} style={{ display: "grid", gridTemplateColumns: "150px minmax(0, 1fr)", gap: 16, padding: "14px 0", borderTop: "1px solid var(--border)" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-start" }}>
                <Badge variant={insight.kind === "recommendation" ? "accent" : "outline"}>{INSIGHT_KIND_LABEL[insight.kind]}</Badge>
                <span style={{ fontSize: 12, color: "var(--text-faint)" }}>{CONFIDENCE_LABEL[insight.confidence]}</span>
              </div>
              <p style={{ fontSize: 14, margin: 0, lineHeight: 1.6 }}>{insight.text}</p>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

function factorsFromObservation(o: TrendObservation): TrendFactors {
  return {
    productRelevance: o.product_relevance,
    commercialIntent: o.commercial_intent,
    regionalMomentum: o.regional_momentum,
    historicalPerformance: o.historical_performance,
    seasonalityTiming: o.seasonality_timing,
    contentSuitability: o.content_suitability,
    inventoryAvailability: o.inventory_availability,
  };
}

function UtmBuilder() {
  const [baseUrl, setBaseUrl] = useState("https://studiostonecreative.com/");
  const [source, setSource] = useState<UtmSource>("instagram");
  const [medium, setMedium] = useState<UtmMedium>("organic-social");
  const [campaign, setCampaign] = useState("");
  const [content, setContent] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  function handleBuild(e: React.FormEvent) {
    e.preventDefault();
    setCopied(false);
    try {
      setResult(buildUtmUrl(baseUrl, { source, medium, campaign, content: content || undefined }));
      setError(null);
    } catch (err) {
      setResult(null);
      setError(err instanceof InvalidUtmUrlError ? err.message : "Unable to build that URL.");
    }
  }

  async function handleCopy() {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  return (
    <Panel
      title="Link builder"
      subtitle="Tag every social link so Shopify can tell you which posts sell"
      about={
        <p>
          Always use these exact source values (instagram, facebook, tiktok, pinterest), never &quot;IG&quot; or &quot;Instagram&quot;, so
          Shopify&apos;s campaign reports group them together.
        </p>
      }
    >
      <form onSubmit={handleBuild} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 14, alignItems: "end" }}>
        <div className="field" style={{ gridColumn: "1 / -1" }}>
          <label htmlFor="utm-url">Destination URL</label>
          <input id="utm-url" className="input" required value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="utm-source">Source</label>
          <select id="utm-source" className="input" value={source} onChange={(e) => setSource(e.target.value as UtmSource)}>
            {UTM_SOURCES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="utm-medium">Medium</label>
          <select id="utm-medium" className="input" value={medium} onChange={(e) => setMedium(e.target.value as UtmMedium)}>
            {UTM_MEDIUMS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="utm-campaign">Campaign</label>
          <input id="utm-campaign" className="input" required placeholder="fall-launch" value={campaign} onChange={(e) => setCampaign(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="utm-content">Content (optional)</label>
          <input id="utm-content" className="input" placeholder="video-a" value={content} onChange={(e) => setContent(e.target.value)} />
        </div>
        <button type="submit" className="btn btn-primary">
          Build link
        </button>
      </form>

      {error && <p style={{ color: "var(--negative)", fontSize: 12.5, marginTop: 12 }}>{error}</p>}

      {result && (
        <div className="card-quiet" style={{ marginTop: 16, padding: 12, display: "flex", alignItems: "center", gap: 12 }}>
          <code className="font-mono" style={{ fontSize: 12, wordBreak: "break-all", flex: 1 }}>
            {result}
          </code>
          <button type="button" className="btn btn-sm" onClick={handleCopy}>
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      )}
    </Panel>
  );
}

const EMPTY_TREND_FORM = {
  term: "",
  platform: "",
  observed_date: "",
  product_relevance: "50",
  commercial_intent: "50",
  regional_momentum: "50",
  historical_performance: "50",
  seasonality_timing: "50",
  content_suitability: "50",
  inventory_availability: "50",
  note: "",
  source_url: "",
};

function TrendObservationForm({ onSaved, onCancel }: { onSaved: () => void; onCancel: () => void }) {
  const [form, setForm] = useState(EMPTY_TREND_FORM);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    const res = await fetch("/api/social-intelligence/trends", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        term: form.term,
        platform: form.platform || null,
        observed_date: form.observed_date,
        product_relevance: Number(form.product_relevance),
        commercial_intent: Number(form.commercial_intent),
        regional_momentum: Number(form.regional_momentum),
        historical_performance: Number(form.historical_performance),
        seasonality_timing: Number(form.seasonality_timing),
        content_suitability: Number(form.content_suitability),
        inventory_availability: Number(form.inventory_availability),
        note: form.note || null,
        source_url: form.source_url || null,
      }),
    });
    setSaving(false);
    if (res.ok) {
      setForm(EMPTY_TREND_FORM);
      onSaved();
    } else {
      const body = await res.json().catch(() => ({}));
      setError(body.error ?? "Unable to save observation.");
    }
  }

  return (
    <form onSubmit={handleSubmit} className="card-quiet" style={{ padding: 16, marginBottom: 18, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 14 }}>
      <div className="field">
        <label htmlFor="trend-term">Term</label>
        <input id="trend-term" className="input" required placeholder="e.g. dragon carving" value={form.term} onChange={(e) => setForm({ ...form, term: e.target.value })} />
      </div>
      <div className="field">
        <label htmlFor="trend-platform">Platform (optional)</label>
        <input id="trend-platform" className="input" placeholder="e.g. pinterest" value={form.platform} onChange={(e) => setForm({ ...form, platform: e.target.value })} />
      </div>
      <div className="field">
        <label htmlFor="trend-date">Observed</label>
        <input id="trend-date" className="input" type="date" required value={form.observed_date} onChange={(e) => setForm({ ...form, observed_date: e.target.value })} />
      </div>
      {FACTOR_FIELDS.map((f) => (
        <div key={f.bodyKey} className="field">
          <label htmlFor={`trend-${f.bodyKey}`}>{f.label} (0–100)</label>
          <input
            id={`trend-${f.bodyKey}`}
            className="input"
            type="number"
            min={0}
            max={100}
            required
            value={form[f.bodyKey as keyof typeof form]}
            onChange={(e) => setForm({ ...form, [f.bodyKey]: e.target.value })}
          />
        </div>
      ))}
      <div className="field" style={{ gridColumn: "span 2" }}>
        <label htmlFor="trend-note">Note</label>
        <input id="trend-note" className="input" value={form.note} onChange={(e) => setForm({ ...form, note: e.target.value })} />
      </div>
      <div className="field">
        <label htmlFor="trend-source">Source URL</label>
        <input id="trend-source" className="input" value={form.source_url} onChange={(e) => setForm({ ...form, source_url: e.target.value })} />
      </div>
      {error && <p style={{ color: "var(--negative)", fontSize: 12.5, margin: 0, gridColumn: "1 / -1" }}>{error}</p>}
      <div style={{ gridColumn: "1 / -1", display: "flex", gap: 8 }}>
        <button type="submit" className="btn btn-primary" disabled={saving}>
          {saving ? "Saving…" : "Save observation"}
        </button>
        <button type="button" className="btn btn-ghost" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}

type TabKey = "planner" | "insights" | "performance" | "trends" | "tools";
const TABS: { key: TabKey; label: string }[] = [
  { key: "planner", label: "Planner" },
  { key: "insights", label: "Insights" },
  { key: "performance", label: "Performance" },
  { key: "trends", label: "Trends" },
  { key: "tools", label: "Link builder" },
];

export default function SocialPlannerPage() {
  const [me, setMe] = useState<Me | null>(null);
  const [posts, setPosts] = useState<SocialPost[]>([]);
  const [postsLoading, setPostsLoading] = useState(true);
  const [attribution, setAttribution] = useState<AttributionResult | null>(null);
  const [attributionLoading, setAttributionLoading] = useState(true);
  const [observations, setObservations] = useState<TrendObservation[]>([]);
  const [observationsLoading, setObservationsLoading] = useState(true);
  const [recommendations, setRecommendations] = useState<TrendRecommendation[]>([]);
  const [recommendationsLoading, setRecommendationsLoading] = useState(true);
  const [ideas, setIdeas] = useState<PlannerIdea[]>([]);
  const [ideasLoading, setIdeasLoading] = useState(true);
  const [occasions, setOccasions] = useState<CalendarOccasion[]>([]);
  const [tradeShows, setTradeShows] = useState<CalendarTradeShow[]>([]);
  const [customItems, setCustomItems] = useState<CalendarCustomItem[]>([]);
  const [calendarLoading, setCalendarLoading] = useState(true);
  const [tab, setTab] = useState<TabKey>("planner");
  const [showTrendForm, setShowTrendForm] = useState(false);

  const loadMe = useCallback(async () => {
    const res = await fetch("/api/auth/me", { cache: "no-store" });
    if (res.ok) setMe((await res.json()).person);
  }, []);

  const loadPosts = useCallback(async () => {
    const res = await fetch("/api/marketing/social", { cache: "no-store" });
    if (res.ok) setPosts((await res.json()).posts);
    setPostsLoading(false);
  }, []);

  const loadAttribution = useCallback(async () => {
    const res = await fetch("/api/social-intelligence/attribution", { cache: "no-store" });
    if (res.ok) setAttribution(await res.json());
    setAttributionLoading(false);
  }, []);

  const loadObservations = useCallback(async () => {
    const res = await fetch("/api/social-intelligence/trends", { cache: "no-store" });
    if (res.ok) setObservations((await res.json()).observations);
    setObservationsLoading(false);
  }, []);

  const loadRecommendations = useCallback(async () => {
    const res = await fetch("/api/social-intelligence/recommendations", { cache: "no-store" });
    if (res.ok) setRecommendations((await res.json()).recommendations);
    setRecommendationsLoading(false);
  }, []);

  const loadIdeas = useCallback(async () => {
    const res = await fetch("/api/social-intelligence/content-ideas", { cache: "no-store" });
    if (res.ok) setIdeas((await res.json()).ideas);
    setIdeasLoading(false);
  }, []);

  const loadCalendar = useCallback(async () => {
    const [occRes, showsRes, itemsRes] = await Promise.all([
      fetch("/api/social-intelligence/occasions", { cache: "no-store" }),
      fetch("/api/social-intelligence/trade-shows", { cache: "no-store" }),
      fetch("/api/social-intelligence/calendar-items", { cache: "no-store" }),
    ]);
    if (occRes.ok) setOccasions((await occRes.json()).occasions);
    if (showsRes.ok) setTradeShows((await showsRes.json()).tradeShows);
    if (itemsRes.ok) setCustomItems((await itemsRes.json()).items);
    setCalendarLoading(false);
  }, []);

  useEffect(() => {
    loadMe();
    loadPosts();
    loadAttribution();
    loadObservations();
    loadRecommendations();
    loadIdeas();
    loadCalendar();
  }, [loadMe, loadPosts, loadAttribution, loadObservations, loadRecommendations, loadIdeas, loadCalendar]);

  // Teammates edit the same planner, so pick up their changes when this tab comes back into view.
  useEffect(() => {
    const onFocus = () => {
      if (document.visibilityState === "visible") {
        loadIdeas();
        loadCalendar();
        loadPosts();
      }
    };
    document.addEventListener("visibilitychange", onFocus);
    return () => document.removeEventListener("visibilitychange", onFocus);
  }, [loadIdeas, loadCalendar, loadPosts]);

  async function handleDeleteObservation(id: number) {
    await fetch(`/api/social-intelligence/trends/${id}`, { method: "DELETE" });
    loadObservations();
  }

  async function handleLogRecommendation(observation: TrendObservation, score: number, classification: string) {
    await fetch("/api/social-intelligence/recommendations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        trend_observation_id: observation.id,
        term: observation.term,
        score_at_recommendation: score,
        classification_at_recommendation: classification,
      }),
    });
    loadRecommendations();
  }

  async function handleLinkRecommendation(id: number, socialPostId: number | null) {
    await fetch(`/api/social-intelligence/recommendations/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ linked_social_post_id: socialPostId }),
    });
    loadRecommendations();
  }

  async function handleDeleteRecommendation(id: number) {
    await fetch(`/api/social-intelligence/recommendations/${id}`, { method: "DELETE" });
    loadRecommendations();
  }

  const scoredObservations = useMemo(
    () => observations.map((o) => ({ observation: o, result: scoreTrendFit(factorsFromObservation(o)) })).sort((a, b) => b.result.score - a.result.score),
    [observations],
  );

  const scores = useMemo(() => {
    const map = new Map<number, { score: number | null; hasRevenueSignal: boolean }>();
    for (const s of scoreSocialPosts(posts.map((p) => ({ id: p.id, revenueAttributed: p.revenue_attributed, linkClicks: p.link_clicks, likes: p.likes, comments: p.comments, shares: p.shares })))) {
      map.set(s.id, s);
    }
    return map;
  }, [posts]);

  const hasAnyRevenueSignal = posts.some((p) => p.revenue_attributed > 0);
  const attributionRows = attribution?.rows ?? [];
  const toReview = ideas.filter((i) => i.status === "suggested").length;

  return (
    <Page title={me ? `Hi, ${me.name.split(" ")[0]}` : "Social planner"} description={toReview > 0 ? `${toReview} content idea${toReview === 1 ? "" : "s"} waiting for review.` : "Everything's reviewed."}>
      <div role="tablist" aria-label="Sections" style={{ display: "flex", gap: 24, borderBottom: "1px solid var(--border)", marginTop: -8, overflowX: "auto" }}>
        {TABS.map((t) => {
          const active = t.key === tab;
          return (
            <button
              key={t.key}
              role="tab"
              aria-selected={active}
              type="button"
              onClick={() => setTab(t.key)}
              style={{
                border: "none",
                background: "none",
                padding: "10px 0",
                marginBottom: -1,
                fontSize: 14,
                fontWeight: active ? 600 : 400,
                color: active ? "var(--text)" : "var(--text-soft)",
                borderBottom: `2px solid ${active ? "var(--accent)" : "transparent"}`,
                cursor: "pointer",
                whiteSpace: "nowrap",
              }}
            >
              {t.label}
            </button>
          );
        })}
      </div>

      {tab === "planner" && (
        <Planner
          ideas={ideas}
          occasions={occasions}
          tradeShows={tradeShows}
          customItems={customItems}
          loggedPosts={posts}
          loading={ideasLoading || calendarLoading}
          me={me}
          // Marking a post as posted writes to the Post log, so refresh both.
          onIdeasChanged={async () => {
            await Promise.all([loadIdeas(), loadPosts()]);
          }}
          onCalendarChanged={async () => {
            await Promise.all([loadCalendar(), loadIdeas(), loadPosts()]);
          }}
        />
      )}

      {tab === "insights" && <AutoInsightsPanel />}

      {tab === "performance" && (
        <>
          <Panel
            title="Sales from social links"
            subtitle="Shopify campaign attribution, last 180 days"
            about={<p>Shopify only credits a campaign when the visitor arrived through a tagged link. Use the Link builder when sharing links from social posts.</p>}
          >
            {attributionLoading ? (
              <p className="muted">Loading…</p>
            ) : !attribution?.configured ? (
              <p className="muted" style={{ fontSize: 13 }}>
                Shopify isn&apos;t connected. Set <code>SHOPIFY_STORE_DOMAIN</code>, <code>SHOPIFY_CLIENT_ID</code> and <code>SHOPIFY_CLIENT_SECRET</code> in the
                repo&apos;s <code>.env</code> to see campaign sales here.
              </p>
            ) : attributionRows.length === 0 ? (
              <p className="muted" style={{ fontSize: 13 }}>
                No tagged visits or orders in the last 180 days. That&apos;s a real result, not an error: tag links with the Link builder and they&apos;ll
                show up here.
              </p>
            ) : (
              <DataTable
                emptyText="No campaign attribution data."
                rows={attributionRows}
                columns={[
                  { header: "Campaign", render: (r) => r.utmCampaign },
                  { header: "Source", render: (r) => r.utmSource },
                  { header: "Medium", render: (r) => r.utmMedium },
                  { header: "Sessions", render: (r) => formatNumber(r.sessions), align: "right" },
                  { header: "Conv. rate", render: (r) => `${(r.conversionRate * 100).toFixed(1)}%`, align: "right" },
                  { header: "Orders", render: (r) => formatNumber(r.orders), align: "right" },
                  { header: "Sales", render: (r) => formatCurrency(r.sales), align: "right" },
                  { header: "AOV", render: (r) => formatCurrency(r.averageOrderValue), align: "right" },
                ]}
              />
            )}
          </Panel>

          <Panel
            title="Post performance"
            subtitle="Every logged post, scored for conversion"
            about={<p>Scores favour revenue first, then link clicks, then engagement. A post with no revenue signal is ranked on traffic only and says so.</p>}
          >
            {!hasAnyRevenueSignal && posts.length > 0 && (
              <p className="pill pill-accent" style={{ marginBottom: 14, height: "auto", padding: "6px 10px", whiteSpace: "normal", borderRadius: 8 }}>
                No post has attributed revenue yet, so these scores rank traffic, not sales.
              </p>
            )}
            {postsLoading ? (
              <p className="muted">Loading…</p>
            ) : (
              <DataTable
                emptyText="No posts logged yet. Add one on the Post log page."
                rows={posts}
                columns={[
                  { header: "Date", render: (p) => formatDate(p.posted_date) },
                  { header: "Platform", render: (p) => <span style={{ textTransform: "capitalize" }}>{p.platform}</span> },
                  { header: "Caption", render: (p) => <span className="muted">{p.caption?.slice(0, 60) ?? "—"}</span> },
                  { header: "Campaign", render: (p) => p.utm_campaign ?? "—" },
                  { header: "Engagement", render: (p) => formatNumber(p.likes + p.comments + p.shares), align: "right" },
                  { header: "Clicks", render: (p) => formatNumber(p.link_clicks), align: "right" },
                  { header: "Revenue", render: (p) => (p.revenue_attributed > 0 ? formatCurrency(p.revenue_attributed) : "—"), align: "right" },
                  {
                    header: "Score",
                    render: (p) => {
                      const s = scores.get(p.id);
                      if (!s || s.score === null) return <Badge>Not enough data</Badge>;
                      return <Badge variant={s.hasRevenueSignal ? "second" : "outline"}>{`${s.score}${!s.hasRevenueSignal ? " · traffic" : ""}`}</Badge>;
                    },
                    align: "right",
                  },
                ]}
              />
            )}
          </Panel>
        </>
      )}

      {tab === "trends" && (
        <>
          <Panel
            title="What should we post next?"
            subtitle="Your top three trend observations, scored for fit"
            about={
              <p>
                Based only on research logged below. No live trend provider is connected, so treat each one as a starting idea to test, not a sure
                thing.
              </p>
            }
          >
            {scoredObservations.length === 0 ? (
              <p className="muted" style={{ fontSize: 13 }}>
                Log a trend observation below to see suggestions here.
              </p>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 280px), 1fr))", gap: 12 }}>
                {scoredObservations.slice(0, 3).map(({ observation, result }) => (
                  <div key={observation.id} className="card-quiet" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 10 }}>
                    <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                      <strong style={{ fontSize: 15, fontWeight: 600, marginRight: "auto" }}>{observation.term}</strong>
                      <span className="font-display tabular" style={{ fontSize: 22 }}>
                        {result.score}
                      </span>
                    </div>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      <Badge variant="accent">{result.classification}</Badge>
                      {observation.platform && <Badge>{observation.platform}</Badge>}
                    </div>
                    <p className="muted" style={{ fontSize: 12.5, margin: 0 }}>
                      {hedgeForClassification(result.classification)}
                    </p>
                    {observation.note && <p style={{ fontSize: 13, margin: 0 }}>{observation.note}</p>}
                    <details className="about">
                      <summary>Factor scores</summary>
                      {FACTOR_FIELDS.map((f) => (
                        <p key={f.bodyKey} style={{ margin: "0 0 2px" }}>
                          {f.label}: {observation[f.bodyKey as keyof TrendObservation]}
                        </p>
                      ))}
                    </details>
                    <button type="button" className="btn btn-sm" style={{ alignSelf: "flex-start" }} onClick={() => handleLogRecommendation(observation, result.score, result.classification)}>
                      Log as recommendation
                    </button>
                  </div>
                ))}
              </div>
            )}
          </Panel>

          <Panel
            title="Trend research"
            subtitle="Your own 0–100 judgement on each factor"
            about={
              <p>
                Each factor is a judgement call, not a live metric. Product relevance matters most: a trend with weak product relevance never scores
                above &quot;Low relevance&quot;, however strong the rest is.
              </p>
            }
            headerAction={
              !showTrendForm && (
                <button type="button" className="btn btn-sm" onClick={() => setShowTrendForm(true)}>
                  Log observation
                </button>
              )
            }
          >
            {showTrendForm && (
              <TrendObservationForm
                onCancel={() => setShowTrendForm(false)}
                onSaved={() => {
                  setShowTrendForm(false);
                  loadObservations();
                }}
              />
            )}
            {observationsLoading ? (
              <p className="muted">Loading…</p>
            ) : (
              <DataTable
                emptyText="No trend observations logged yet."
                rows={scoredObservations}
                columns={[
                  { header: "Date", render: (row) => formatDate(row.observation.observed_date) },
                  { header: "Term", render: (row) => row.observation.term },
                  { header: "Platform", render: (row) => row.observation.platform ?? "—" },
                  { header: "Fit score", render: (row) => String(row.result.score), align: "right" },
                  { header: "Classification", render: (row) => <Badge variant={row.result.classification === "Ignore" || row.result.classification === "Low Relevance" ? "outline" : "accent"}>{row.result.classification}</Badge> },
                  {
                    header: "",
                    render: (row) => (
                      <button type="button" className="link-btn danger" onClick={() => handleDeleteObservation(row.observation.id)}>
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
            title="Did it work?"
            subtitle="Recommendations compared with what actually happened"
            about={
              <p>
                Link each recommendation to the post that came out of it to see the real result. One linked post is never enough to draw a conclusion,
                and each comparison says so.
              </p>
            }
          >
            {recommendationsLoading ? (
              <p className="muted">Loading…</p>
            ) : (
              <DataTable
                emptyText='No recommendations logged yet. Use "Log as recommendation" above.'
                rows={recommendations}
                columns={[
                  { header: "Recommended", render: (r) => formatDate(r.recommended_date) },
                  { header: "Term", render: (r) => r.term },
                  { header: "Predicted", render: (r) => <Badge>{`${r.classification_at_recommendation} · ${r.score_at_recommendation}`}</Badge> },
                  {
                    header: "Linked post",
                    render: (r) => (
                      <select
                        className="input"
                        aria-label={`Linked post for ${r.term}`}
                        value={r.linked_social_post_id ?? ""}
                        onChange={(e) => handleLinkRecommendation(r.id, e.target.value ? Number(e.target.value) : null)}
                        style={{ width: "auto", maxWidth: 260 }}
                      >
                        <option value="">Not linked</option>
                        {posts.map((p) => (
                          <option key={p.id} value={p.id}>
                            {formatDate(p.posted_date)} · {p.platform} · {p.caption?.slice(0, 30) ?? `post #${p.id}`}
                          </option>
                        ))}
                      </select>
                    ),
                  },
                  {
                    header: "Outcome",
                    render: (r) => {
                      const linkedPost = posts.find((p) => p.id === r.linked_social_post_id) ?? null;
                      const outcome = evaluateLearningLoopOutcome(
                        linkedPost
                          ? { revenueAttributed: linkedPost.revenue_attributed, linkClicks: linkedPost.link_clicks, likes: linkedPost.likes, comments: linkedPost.comments, shares: linkedPost.shares }
                          : null,
                      );
                      const label = { converted: "Converted", "engaged-no-revenue": "Engaged, no revenue", "no-engagement": "No engagement", "awaiting-outcome": "Waiting" }[outcome.outcome];
                      return (
                        <span title={outcome.note}>
                          <Badge variant={outcome.outcome === "converted" ? "second" : "outline"}>{label}</Badge>
                        </span>
                      );
                    },
                  },
                  {
                    header: "",
                    render: (r) => (
                      <button type="button" className="link-btn danger" onClick={() => handleDeleteRecommendation(r.id)}>
                        Delete
                      </button>
                    ),
                    align: "right",
                  },
                ]}
              />
            )}
          </Panel>

          <Panel title="Trend data sources" subtitle="Which live trend feeds are connected">
            <div style={{ display: "flex", flexDirection: "column" }}>
              {TREND_PROVIDERS.map((p) => (
                <div key={p.id} style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto", gap: 12, padding: "12px 0", borderTop: "1px solid var(--border)" }}>
                  <div>
                    <div style={{ fontSize: 13.5, fontWeight: 500 }}>{p.label}</div>
                    <div className="muted" style={{ fontSize: 12.5 }}>
                      {p.note}
                    </div>
                  </div>
                  <Badge variant={p.status === "connected" ? "second" : "outline"}>{p.status === "connected" ? "Connected" : "Not connected"}</Badge>
                </div>
              ))}
            </div>
          </Panel>
        </>
      )}

      {tab === "tools" && <UtmBuilder />}
    </Page>
  );
}
