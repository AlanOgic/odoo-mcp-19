import "server-only";

import { getHealthDb } from "../db";
import { createSocialPost, PLANNER_LOG_NOTE, type SocialPlatform } from "./socialPosts";

/**
 * Imports two kinds of Meta Business Suite CSV — real numbers, never fabricated:
 *
 * 1. Per-post exports (Insights → Content → Export), one file per platform:
 *    a Facebook Page file ("Page ID", "Reactions", "Total clicks", …) or an
 *    Instagram file ("Account username", "Likes", "Saves", …), one row per
 *    post with lifetime totals. Each row updates the matching Post log entry
 *    or adds one — see importMetaPostExport.
 *
 * 2. The long-format summary export (one row per
 *    platform/record_type/metric/date). Two kinds of rows are used:
 *   - record_type "post": pivoted into social_posts rows (one row per
 *     platform per post), matching the same schema the Social Media page's
 *     manual-entry form writes to.
 *   - record_type "summary": kept as-is in social_platform_insights (one row
 *     per platform/metric/period) rather than normalized into fixed columns,
 *     since the metric set differs per platform and grows over time.
 * "daily" rows are intentionally skipped — see the social_platform_insights
 * schema comment in lib/db.ts.
 */

const SUPPORTED_PLATFORMS = new Set<SocialPlatform>(["instagram", "facebook", "tiktok", "pinterest"]);

/** Splits CSV text into records of fields. Quoted fields may contain commas,
 * doubled quotes and line breaks — Meta's post exports put multi-line captions
 * in quotes, so the text can't be split on newlines first. */
function parseCsvRecords(text: string): string[][] {
  const records: string[][] = [];
  let record: string[] = [];
  let field = "";
  let inQuotes = false;
  const src = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < src.length; i++) {
    const ch = src[i];
    if (inQuotes) {
      if (ch === '"') {
        if (src[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      record.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && src[i + 1] === "\n") i++;
      record.push(field);
      records.push(record);
      record = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field !== "" || record.length > 0) {
    record.push(field);
    records.push(record);
  }
  return records.filter((r) => r.some((f) => f.trim() !== ""));
}

function parseCsv(text: string): Record<string, string>[] {
  const records = parseCsvRecords(text);
  if (records.length === 0) return [];
  const header = records[0]!.map((h) => h.trim());
  return records.slice(1).map((fields) => {
    const row: Record<string, string> = {};
    header.forEach((h, idx) => {
      row[h] = (fields[idx] ?? "").trim();
    });
    return row;
  });
}

const POST_METRIC_TO_FIELD: Record<string, "reach" | "likes" | "comments" | "shares"> = {
  Views: "reach",
  Reach: "reach",
  "Reactions/Likes": "likes",
  Comments: "comments",
  Shares: "shares",
};

export interface ParsedPost {
  platform: SocialPlatform;
  postedDate: string;
  title: string;
  reach: number | null;
  likes: number;
  comments: number;
  shares: number;
}

export interface ParsedSummary {
  platform: SocialPlatform;
  metric: string;
  periodStart: string;
  periodEnd: string;
  value: number | null;
  unit: string | null;
  changePct: number | null;
}

export interface ParsedSocialInsights {
  posts: ParsedPost[];
  summaries: ParsedSummary[];
}

function toNumber(raw: string | undefined): number | null {
  if (!raw) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

export function parseSocialInsightsCsv(csvText: string): ParsedSocialInsights {
  const rows = parseCsv(csvText);
  const summaries: ParsedSummary[] = [];
  const postsByKey = new Map<string, ParsedPost>();

  for (const row of rows) {
    const platform = row.platform?.toLowerCase() as SocialPlatform | undefined;
    if (!platform || !SUPPORTED_PLATFORMS.has(platform)) continue;

    if (row.record_type === "summary") {
      summaries.push({
        platform,
        metric: row.metric ?? "",
        periodStart: row.period_start ?? "",
        periodEnd: row.period_end ?? "",
        value: toNumber(row.value),
        unit: row.unit || null,
        changePct: toNumber(row.change_vs_prev_period_pct),
      });
    } else if (row.record_type === "post") {
      const key = `${platform}|${row.post_published}|${row.post_title}`;
      const existing: ParsedPost = postsByKey.get(key) ?? {
        platform,
        postedDate: row.date ?? "",
        title: row.post_title ?? "",
        reach: null,
        likes: 0,
        comments: 0,
        shares: 0,
      };
      const field = row.metric ? POST_METRIC_TO_FIELD[row.metric] : undefined;
      const value = toNumber(row.value);
      if (field === "reach") existing.reach = value;
      else if (field === "likes") existing.likes = value ?? 0;
      else if (field === "comments") existing.comments = value ?? 0;
      else if (field === "shares") existing.shares = value ?? 0;
      postsByKey.set(key, existing);
    }
  }

  return { posts: Array.from(postsByKey.values()), summaries };
}

export interface ImportSocialInsightsResult {
  postsImported: number;
  /** Existing Post log entries whose numbers were refreshed (per-post exports). */
  postsUpdated: number;
  postsSkipped: number;
  summariesImported: number;
}

export interface ParsedPostExportRow {
  platform: SocialPlatform;
  externalId: string;
  postedDate: string;
  caption: string;
  postType: string | null;
  permalink: string | null;
  views: number | null;
  reach: number | null;
  likes: number;
  comments: number;
  shares: number;
  /** Facebook exports have clicks; Instagram's don't, so null means "not in this file". */
  linkClicks: number | null;
}

/** "09/28/2026 18:12" → "2026-09-28". */
function publishDate(raw: string): string | null {
  const m = raw.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})/);
  return m ? `${m[3]}-${m[1]!.padStart(2, "0")}-${m[2]!.padStart(2, "0")}` : null;
}

/**
 * Parses a Meta per-post export, or returns null if the CSV isn't one. The
 * platform is told apart by its columns: Facebook Page exports have "Page ID",
 * Instagram ones "Account username".
 */
export function parseMetaPostExport(csvText: string): ParsedPostExportRow[] | null {
  const rows = parseCsv(csvText);
  const first = rows[0];
  if (!first || !("Post ID" in first) || !("Publish time" in first)) return null;
  const platform: SocialPlatform | null = "Page ID" in first ? "facebook" : "Account username" in first ? "instagram" : null;
  if (!platform) return null;

  const out: ParsedPostExportRow[] = [];
  for (const row of rows) {
    const postedDate = publishDate(row["Publish time"] ?? "");
    if (!row["Post ID"] || !postedDate) continue;
    const num = (key: string) => toNumber(row[key]);
    out.push({
      platform,
      externalId: row["Post ID"],
      postedDate,
      caption: (platform === "facebook" ? row.Title || row.Description : row.Description || row.Title) ?? "",
      postType: row["Post type"] || null,
      permalink: row.Permalink || null,
      views: num("Views"),
      reach: num("Reach"),
      likes: (platform === "facebook" ? num("Reactions") : num("Likes")) ?? 0,
      comments: num("Comments") ?? 0,
      shares: num("Shares") ?? 0,
      // Facebook's "Other Clicks" are clicks on the post other than opening the
      // photo — the closest the Page export gets to link clicks.
      linkClicks: platform === "facebook" ? (num("Link clicks") ?? num("Other Clicks")) : null,
    });
  }
  return out;
}

/** Lowercased, whitespace-collapsed opening of a caption, for matching posts
 * imported before Post IDs were stored. */
function captionKey(caption: string | null): string {
  return (caption ?? "").toLowerCase().replace(/\s+/g, " ").trim().slice(0, 40);
}

/**
 * Upserts a per-post export into the Post log. A row matches an existing entry
 * by Meta's Post ID, then permalink, then same platform + day + caption opening
 * (entries imported before IDs were kept), then an entry the planner logged for
 * that platform and day without numbers yet. Matches get the export's lifetime
 * numbers; anything else is added.
 */
export function importMetaPostExport(rows: ParsedPostExportRow[]): { postsImported: number; postsUpdated: number } {
  const db = getHealthDb();
  const byExternalId = db.prepare("SELECT id FROM social_posts WHERE external_id = ?");
  const byPermalink = db.prepare("SELECT id FROM social_posts WHERE permalink = ?");
  const sameDay = db.prepare("SELECT id, caption, notes, likes, comments, shares, reach FROM social_posts WHERE platform = ? AND posted_date = ? AND external_id IS NULL ORDER BY id");
  const update = db.prepare(`
    UPDATE social_posts SET external_id = @externalId, permalink = @permalink, post_type = @postType, caption = @caption,
      views = @views, reach = @reach, likes = @likes, comments = @comments, shares = @shares,
      link_clicks = COALESCE(@linkClicks, link_clicks),
      notes = CASE WHEN notes = @plannerNote THEN 'Logged from the planner; numbers from a Meta post export.' ELSE notes END
    WHERE id = @id`);

  let postsImported = 0;
  let postsUpdated = 0;
  db.transaction(() => {
    for (const r of rows) {
      let id = (byExternalId.get(r.externalId) as { id: number } | undefined)?.id ?? (r.permalink ? (byPermalink.get(r.permalink) as { id: number } | undefined)?.id : undefined);
      if (id === undefined) {
        const candidates = sameDay.all(r.platform, r.postedDate) as { id: number; caption: string | null; notes: string | null; likes: number; comments: number; shares: number; reach: number | null }[];
        id =
          candidates.find((c) => captionKey(c.caption) !== "" && captionKey(c.caption) === captionKey(r.caption))?.id ??
          candidates.find((c) => c.notes === PLANNER_LOG_NOTE && c.likes + c.comments + c.shares === 0 && c.reach === null)?.id;
      }
      if (id !== undefined) {
        update.run({ ...r, id, plannerNote: PLANNER_LOG_NOTE });
        postsUpdated++;
        continue;
      }
      const created = createSocialPost({
        posted_date: r.postedDate,
        platform: r.platform,
        post_type: r.postType,
        caption: r.caption || null,
        likes: r.likes,
        comments: r.comments,
        shares: r.shares,
        link_clicks: r.linkClicks ?? 0,
        reach: r.reach,
        notes: "Imported from a Meta post export.",
      });
      db.prepare("UPDATE social_posts SET external_id = ?, permalink = ?, views = ? WHERE id = ?").run(r.externalId, r.permalink, r.views, created.id);
      postsImported++;
    }
  })();
  return { postsImported, postsUpdated };
}

/** Idempotent: a post already present (matched on platform + posted_date + caption)
 * is skipped rather than duplicated; summary rows upsert on their unique
 * (platform, metric, period_start, period_end) key, so re-importing a corrected
 * export overwrites rather than duplicating. */
export function importSocialInsightsCsv(csvText: string): ImportSocialInsightsResult {
  const postExport = parseMetaPostExport(csvText);
  if (postExport) return { ...importMetaPostExport(postExport), postsSkipped: 0, summariesImported: 0 };

  const { posts, summaries } = parseSocialInsightsCsv(csvText);
  const db = getHealthDb();

  let postsImported = 0;
  let postsSkipped = 0;
  const existsPost = db.prepare(`SELECT id FROM social_posts WHERE platform = ? AND posted_date = ? AND caption = ?`);
  // An entry the planner logged for this platform and day, still without numbers,
  // is the same post: fill it in rather than adding a second row.
  const plannerEntry = db.prepare(
    `SELECT id FROM social_posts WHERE platform = ? AND posted_date = ? AND notes = ? AND likes = 0 AND comments = 0 AND shares = 0 AND reach IS NULL ORDER BY id LIMIT 1`,
  );
  const fillPlannerEntry = db.prepare(
    `UPDATE social_posts SET caption = ?, likes = ?, comments = ?, shares = ?, reach = ?, notes = ? WHERE id = ?`,
  );
  for (const p of posts) {
    if (existsPost.get(p.platform, p.postedDate, p.title)) {
      postsSkipped++;
      continue;
    }
    const logged = plannerEntry.get(p.platform, p.postedDate, PLANNER_LOG_NOTE) as { id: number } | undefined;
    if (logged) {
      fillPlannerEntry.run(p.title, p.likes, p.comments, p.shares, p.reach, "Imported from a Meta Business Suite insights export (matched a post logged from the planner).", logged.id);
      postsImported++;
      continue;
    }
    createSocialPost({
      posted_date: p.postedDate,
      platform: p.platform,
      caption: p.title,
      likes: p.likes,
      comments: p.comments,
      shares: p.shares,
      link_clicks: 0,
      reach: p.reach,
      notes: "Imported from a Meta Business Suite insights export.",
    });
    postsImported++;
  }

  const upsertSummary = db.prepare(`
    INSERT INTO social_platform_insights (platform, metric, period_start, period_end, value, unit, change_vs_prev_period_pct)
    VALUES (@platform, @metric, @periodStart, @periodEnd, @value, @unit, @changePct)
    ON CONFLICT(platform, metric, period_start, period_end) DO UPDATE SET
      value = excluded.value,
      unit = excluded.unit,
      change_vs_prev_period_pct = excluded.change_vs_prev_period_pct,
      imported_at = datetime('now')
  `);
  for (const s of summaries) {
    upsertSummary.run(s);
  }

  return { postsImported, postsUpdated: 0, postsSkipped, summariesImported: summaries.length };
}

export interface PlatformInsightRow {
  platform: SocialPlatform;
  metric: string;
  period_start: string;
  period_end: string;
  value: number | null;
  unit: string | null;
  change_vs_prev_period_pct: number | null;
}

/** The most recently-imported period's summary metrics for each platform (not
 * necessarily the same period for every platform, if imports happen at
 * different times) — each platform's own latest period_end wins. */
export function listLatestPlatformInsights(): PlatformInsightRow[] {
  return getHealthDb()
    .prepare(
      `SELECT platform, metric, period_start, period_end, value, unit, change_vs_prev_period_pct
       FROM social_platform_insights
       WHERE (platform, period_end) IN (
         SELECT platform, MAX(period_end) FROM social_platform_insights GROUP BY platform
       )
       ORDER BY platform ASC, metric ASC`,
    )
    .all() as PlatformInsightRow[];
}
