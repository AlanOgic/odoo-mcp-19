import "server-only";

import { getHealthDb } from "../db";

export type SocialPlatform = "instagram" | "facebook" | "tiktok" | "pinterest";

/** Marks a Post log entry the planner created when a post was marked as posted.
 * A later Meta import for the same platform and day fills in that entry's
 * numbers instead of adding a duplicate (socialInsightsImport.ts). */
export const PLANNER_LOG_NOTE = "Logged from the planner.";

export interface SocialPost {
  id: number;
  posted_date: string;
  platform: SocialPlatform;
  post_type: string | null;
  caption: string | null;
  likes: number;
  comments: number;
  shares: number;
  link_clicks: number;
  reach: number | null;
  views: number | null;
  permalink: string | null;
  external_id: string | null;
  notes: string | null;
  utm_source: string | null;
  utm_medium: string | null;
  utm_campaign: string | null;
  revenue_attributed: number;
  created_at: string;
}

export interface NewSocialPost {
  posted_date: string;
  platform: SocialPlatform;
  post_type?: string | null;
  caption?: string | null;
  likes: number;
  comments: number;
  shares: number;
  link_clicks: number;
  reach?: number | null;
  notes?: string | null;
  utm_source?: string | null;
  utm_medium?: string | null;
  utm_campaign?: string | null;
  revenue_attributed?: number;
}

export function listSocialPosts(platform?: SocialPlatform): SocialPost[] {
  const db = getHealthDb();
  if (platform) {
    return db.prepare("SELECT * FROM social_posts WHERE platform = ? ORDER BY posted_date DESC, id DESC").all(platform) as SocialPost[];
  }
  return db.prepare("SELECT * FROM social_posts ORDER BY posted_date DESC, id DESC").all() as SocialPost[];
}

export function createSocialPost(data: NewSocialPost): SocialPost {
  const db = getHealthDb();
  const result = db
    .prepare(
      `INSERT INTO social_posts (posted_date, platform, post_type, caption, likes, comments, shares, link_clicks, reach, notes, utm_source, utm_medium, utm_campaign, revenue_attributed)
       VALUES (@posted_date, @platform, @post_type, @caption, @likes, @comments, @shares, @link_clicks, @reach, @notes, @utm_source, @utm_medium, @utm_campaign, @revenue_attributed)`,
    )
    .run({
      ...data,
      post_type: data.post_type ?? null,
      caption: data.caption ?? null,
      reach: data.reach ?? null,
      notes: data.notes ?? null,
      utm_source: data.utm_source ?? null,
      utm_medium: data.utm_medium ?? null,
      utm_campaign: data.utm_campaign ?? null,
      revenue_attributed: data.revenue_attributed ?? 0,
    });
  return db.prepare("SELECT * FROM social_posts WHERE id = ?").get(result.lastInsertRowid) as SocialPost;
}

export function deleteSocialPost(id: number): boolean {
  const db = getHealthDb();
  const result = db.prepare("DELETE FROM social_posts WHERE id = ?").run(id);
  return result.changes > 0;
}
