import "server-only";

import Database from "better-sqlite3";
import { mkdirSync } from "node:fs";
import path from "node:path";

/**
 * Local SQLite store for the Social Media dashboard. Split out of the
 * Website Health dashboard's monolithic db.ts (dashboard/src/lib/db.ts) —
 * this file keeps only the tables the social features actually use. Never
 * written back to Odoo or Shopify.
 */

let db: Database.Database | null = null;
let currentDbPath: string | null = null;

export function getHealthDb(): Database.Database {
  const dataDir = process.env.DASHBOARD_DATA_DIR ?? path.resolve(process.cwd(), "data");
  const dbPath = path.join(dataDir, "social.sqlite3");

  // If the data directory changed, close the old database
  if (currentDbPath !== dbPath) {
    if (db) {
      db.close();
      db = null;
    }
    currentDbPath = dbPath;
  }

  if (db) return db;
  mkdirSync(dataDir, { recursive: true });
  db = new Database(dbPath);
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");
  db.exec(SCHEMA);
  runMigrations(db);
  return db;
}

/**
 * SQLite's `ALTER TABLE ... ADD COLUMN` has no `IF NOT EXISTS` form, so
 * additive schema changes to a table that may already exist (with data)
 * are applied here, guarded by a PRAGMA table_info check, rather than
 * folded into the idempotent CREATE TABLE statements above.
 */
function runMigrations(db: Database.Database): void {
  dropAccessCodes(db);
  allowCalendarItemActivity(db);
  addColumnIfMissing(db, "social_posts", "utm_source", "TEXT");
  addColumnIfMissing(db, "social_posts", "utm_medium", "TEXT");
  addColumnIfMissing(db, "social_posts", "utm_campaign", "TEXT");
  addColumnIfMissing(db, "social_posts", "revenue_attributed", "REAL NOT NULL DEFAULT 0");
  addColumnIfMissing(db, "social_posts", "reach", "INTEGER");
  // Meta per-post exports: a stable Post ID to match re-imports on (Facebook's
  // permalinks change between exports), the permalink itself, and view counts.
  addColumnIfMissing(db, "social_posts", "external_id", "TEXT");
  addColumnIfMissing(db, "social_posts", "permalink", "TEXT");
  addColumnIfMissing(db, "social_posts", "views", "INTEGER");
  addColumnIfMissing(db, "trade_shows", "lead_days", "INTEGER NOT NULL DEFAULT 10");
  addColumnIfMissing(db, "content_ideas", "format", "TEXT NOT NULL DEFAULT 'photo'");
  addColumnIfMissing(db, "content_ideas", "suggested_time", "TEXT");
  // The Post log entry a posted planner item corresponds to (see lib/social/postLink.ts).
  addColumnIfMissing(db, "content_ideas", "posted_post_id", "INTEGER REFERENCES social_posts(id) ON DELETE SET NULL");
}

/**
 * People briefly signed in with access codes (people.code_hash, role,
 * revoked_at). Codes were dropped for name-only identity; SQLite can't drop a
 * UNIQUE column in place, so rebuild the table once, keeping ids so sessions,
 * comments and history still point at the right person.
 */
function dropAccessCodes(db: Database.Database): void {
  const columns = db.prepare("PRAGMA table_info(people)").all() as Array<{ name: string }>;
  if (!columns.some((c) => c.name === "code_hash")) return;
  db.pragma("foreign_keys = OFF");
  try {
    db.transaction(() => {
      db.exec(`
        CREATE TABLE people_new (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL UNIQUE COLLATE NOCASE,
          created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT OR IGNORE INTO people_new (id, name, created_at) SELECT id, name, created_at FROM people ORDER BY id;
        DROP TABLE people;
        ALTER TABLE people_new RENAME TO people;
      `);
    })();
  } finally {
    db.pragma("foreign_keys = ON");
  }
}

/** activity_log's entity_type CHECK predates calendar_items; SQLite can't alter
 * a CHECK in place, so rebuild the table once with the wider list, keeping rows. */
function allowCalendarItemActivity(db: Database.Database): void {
  const row = db.prepare("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'activity_log'").get() as { sql: string } | undefined;
  if (!row || row.sql.includes("'calendar_item'")) return;
  db.transaction(() => {
    db.exec(`
      CREATE TABLE activity_log_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
        author_name TEXT NOT NULL,
        entity_type TEXT NOT NULL CHECK(entity_type IN ('content_idea','trade_show','social_post','trend','calendar_item')),
        entity_id INTEGER NOT NULL,
        action TEXT NOT NULL,
        summary TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      );
      INSERT INTO activity_log_new SELECT id, person_id, author_name, entity_type, entity_id, action, summary, created_at FROM activity_log;
      DROP TABLE activity_log;
      ALTER TABLE activity_log_new RENAME TO activity_log;
      CREATE INDEX IF NOT EXISTS idx_activity_entity ON activity_log(entity_type, entity_id);
    `);
  })();
}

function addColumnIfMissing(db: Database.Database, table: string, column: string, definition: string): void {
  const columns = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>;
  if (columns.some((c) => c.name === column)) return;
  db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition}`);
}

export function closeHealthDb(): void {
  db?.close();
  db = null;
  currentDbPath = null;
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS social_posts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  posted_date TEXT NOT NULL,
  platform TEXT NOT NULL CHECK(platform IN ('instagram','facebook','tiktok','pinterest')),
  post_type TEXT,
  caption TEXT,
  likes INTEGER NOT NULL DEFAULT 0,
  comments INTEGER NOT NULL DEFAULT 0,
  shares INTEGER NOT NULL DEFAULT 0,
  link_clicks INTEGER NOT NULL DEFAULT 0,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_social_posts_date ON social_posts(posted_date);
CREATE INDEX IF NOT EXISTS idx_social_posts_platform ON social_posts(platform);

-- Social & Conversion Intelligence, Phase 2: manually-logged trend research.
-- Every factor is a person's own 0-100 judgment call, not a live trend-provider
-- metric -- there is no Google Trends/Pinterest Trends/TikTok Creative Center
-- integration yet (Phase 3, gated on API access this project doesn't have).
-- Column names mirror the 7-factor Trend Fit Score model documented in the
-- studiostone-social-conversion-analyst skill's references/scoring-models.md;
-- keep them in sync if that model's factors ever change.
CREATE TABLE IF NOT EXISTS trend_observations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  term TEXT NOT NULL,
  platform TEXT,
  observed_date TEXT NOT NULL,
  product_relevance INTEGER NOT NULL CHECK(product_relevance BETWEEN 0 AND 100),
  commercial_intent INTEGER NOT NULL CHECK(commercial_intent BETWEEN 0 AND 100),
  regional_momentum INTEGER NOT NULL CHECK(regional_momentum BETWEEN 0 AND 100),
  historical_performance INTEGER NOT NULL CHECK(historical_performance BETWEEN 0 AND 100),
  seasonality_timing INTEGER NOT NULL CHECK(seasonality_timing BETWEEN 0 AND 100),
  content_suitability INTEGER NOT NULL CHECK(content_suitability BETWEEN 0 AND 100),
  inventory_availability INTEGER NOT NULL CHECK(inventory_availability BETWEEN 0 AND 100),
  note TEXT,
  source_url TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_trend_observations_date ON trend_observations(observed_date);

-- Social & Conversion Intelligence, Phase 3 (partial): the Learning Loop.
-- Real live trend providers (Google/Pinterest/TikTok) are still not connected --
-- that half of Phase 3 is genuinely blocked on API access this project doesn't
-- have, and building fake adapters for it would violate "never fabricate trend
-- data." What IS real and buildable now: recording what a Trend Fit Score
-- predicted at the moment it was acted on, then -- once a real social_posts row
-- exists for it -- comparing that prediction against real, measured outcomes.
-- A single linked post is not a sample size the spec allows drawing conclusions
-- from (see the skill's sample-size discount table); this table just makes that
-- comparison possible to observe over time, it doesn't itself claim significance.
CREATE TABLE IF NOT EXISTS trend_recommendations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trend_observation_id INTEGER REFERENCES trend_observations(id) ON DELETE SET NULL,
  term TEXT NOT NULL,
  score_at_recommendation INTEGER NOT NULL,
  classification_at_recommendation TEXT NOT NULL,
  recommended_date TEXT NOT NULL DEFAULT (datetime('now')),
  linked_social_post_id INTEGER REFERENCES social_posts(id) ON DELETE SET NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_trend_recommendations_date ON trend_recommendations(recommended_date);

-- Platform-level (not per-post) social insights imported verbatim from exported
-- CSVs (Meta Business Suite / SocialBee-style exports). Kept as one row per
-- (platform, metric, period) rather than normalized into fixed columns, because
-- the set of metrics each platform reports differs and grows over time -- see
-- lib/marketing/socialInsightsImport.ts. UNIQUE constraint makes re-importing
-- the same period idempotent (a corrected re-export overwrites, never duplicates).
CREATE TABLE IF NOT EXISTS social_platform_insights (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  platform TEXT NOT NULL CHECK(platform IN ('instagram','facebook','tiktok','pinterest')),
  metric TEXT NOT NULL,
  period_start TEXT NOT NULL,
  period_end TEXT NOT NULL,
  value REAL,
  unit TEXT,
  change_vs_prev_period_pct REAL,
  imported_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(platform, metric, period_start, period_end)
);
CREATE INDEX IF NOT EXISTS idx_social_platform_insights_period ON social_platform_insights(period_start, period_end);

-- Content ideas (repost/refresh/new) for the Social Media workflow, authored by
-- Claude acting as the studiostone-social-conversion-analyst skill against real
-- performance (social_posts) and real Shopify inventory data -- never a live
-- LLM call at runtime, and never fabricated performance/trend numbers. An idea
-- can optionally reference the real post it's a repost/refresh of, and/or a
-- real upcoming occasion (lib/social/seasonalCalendar.ts) it's timed for.
-- occasion_id is a plain string key from that static list, not a DB foreign
-- key, since occasions are computed rather than stored rows.
CREATE TABLE IF NOT EXISTS content_ideas (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  idea_type TEXT NOT NULL CHECK(idea_type IN ('repost','refresh','new')),
  source_post_id INTEGER REFERENCES social_posts(id) ON DELETE SET NULL,
  occasion_id TEXT,
  target_date TEXT,
  platform TEXT NOT NULL CHECK(platform IN ('instagram','facebook','tiktok','pinterest')),
  product TEXT NOT NULL,
  product_handle TEXT,
  pillar TEXT,
  hook TEXT,
  caption TEXT NOT NULL,
  hashtags TEXT,
  cta TEXT,
  reasoning TEXT NOT NULL,
  confidence TEXT NOT NULL CHECK(confidence IN ('high','promising','experimental','insufficient')),
  inventory_verified INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'suggested' CHECK(status IN ('suggested','approved','used','dismissed')),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_content_ideas_status ON content_ideas(status);
CREATE INDEX IF NOT EXISTS idx_content_ideas_target_date ON content_ideas(target_date);

-- Trade shows Studiostone is actually attending -- distinct from
-- lib/social/seasonalCalendar.ts's universal, code-computed retail dates
-- (Halloween, Black Friday, ...). This calendar is this specific business's
-- travel/exhibition schedule, so it's user-entered data in the DB rather than
-- hardcoded, the same way trend_observations is.
CREATE TABLE IF NOT EXISTS trade_shows (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  location TEXT,
  start_date TEXT NOT NULL,
  end_date TEXT,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_trade_shows_start_date ON trade_shows(start_date);

-- Who's using the planner (LAN sharing, no sign-in): each device picks a name
-- once. Names are unique ignoring case, so "nicole" and "Nicole" are one person.
CREATE TABLE IF NOT EXISTS people (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE COLLATE NOCASE,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Browser sessions, keyed by a SHA-256 of the cookie token.
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY,
  person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_person ON sessions(person_id);

-- Comment threads on scheduled posts (content ideas). author_name is a
-- snapshot so a thread still reads correctly after someone is renamed.
CREATE TABLE IF NOT EXISTS post_comments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content_idea_id INTEGER NOT NULL REFERENCES content_ideas(id) ON DELETE CASCADE,
  person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
  author_name TEXT NOT NULL,
  body TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  edited_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_post_comments_idea ON post_comments(content_idea_id);

-- Who did what: post edits, status changes, comments, event changes.
CREATE TABLE IF NOT EXISTS activity_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
  author_name TEXT NOT NULL,
  entity_type TEXT NOT NULL CHECK(entity_type IN ('content_idea','trade_show','social_post','trend','calendar_item')),
  entity_id INTEGER NOT NULL,
  action TEXT NOT NULL,
  summary TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_activity_entity ON activity_log(entity_type, entity_id);

-- Holidays and post-by deadlines people add themselves on the calendar. The
-- standard retail holidays and the deadlines before them are computed in code
-- (lib/social/seasonalCalendar.ts); these are the extra, studio-specific ones.
CREATE TABLE IF NOT EXISTS calendar_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL CHECK(kind IN ('holiday','deadline')),
  title TEXT NOT NULL,
  date TEXT NOT NULL,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_calendar_items_date ON calendar_items(date);
`;
