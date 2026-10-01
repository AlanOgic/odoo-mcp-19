import "server-only";

import { getHealthDb } from "../db";
import { createSocialPost, PLANNER_LOG_NOTE, type SocialPlatform } from "../marketing/socialPosts";

/**
 * Keeps the planner (content_ideas) and the Post log (social_posts) in step.
 * A planner post marked as posted is linked to the Post log entry for it
 * (content_ideas.posted_post_id): an existing entry for the same platform and
 * day if there is one — e.g. from a Meta import — otherwise a new entry the
 * planner creates, which a later Meta import fills in with real numbers.
 */

interface IdeaRow {
  id: number;
  status: string;
  target_date: string | null;
  platform: SocialPlatform;
  format: string;
  hook: string | null;
  caption: string;
  posted_post_id: number | null;
}

interface PostRow {
  id: number;
  notes: string | null;
  likes: number;
  comments: number;
  shares: number;
  link_clicks: number;
  reach: number | null;
  revenue_attributed: number;
}

const FORMAT_POST_TYPE: Record<string, string> = { photo: "Photo", reel: "Reel", carousel: "Carousel", story: "Story" };

function localToday(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function idea(id: number): IdeaRow | undefined {
  return getHealthDb().prepare("SELECT id, status, target_date, platform, format, hook, caption, posted_post_id FROM content_ideas WHERE id = ?").get(id) as IdeaRow | undefined;
}

function post(id: number): PostRow | undefined {
  return getHealthDb().prepare("SELECT id, notes, likes, comments, shares, link_clicks, reach, revenue_attributed FROM social_posts WHERE id = ?").get(id) as PostRow | undefined;
}

const plannerCreated = (p: PostRow) => p.notes === PLANNER_LOG_NOTE;
const hasNumbers = (p: PostRow) => p.likes + p.comments + p.shares + p.link_clicks > 0 || p.reach !== null || p.revenue_attributed > 0;

function postText(i: IdeaRow): string {
  return [i.hook, i.caption].filter((t) => t && t.trim()).join("\n\n");
}

/** Marks-as-posted side effects: gives an undated post today's date, then links
 * (or creates) its Post log entry. Returns the linked post id. */
export function linkPostedIdea(ideaId: number): number | null {
  const db = getHealthDb();
  let i = idea(ideaId);
  if (!i || i.status !== "used") return null;
  if (!i.target_date) {
    db.prepare("UPDATE content_ideas SET target_date = ? WHERE id = ?").run(localToday(), ideaId);
    i = idea(ideaId)!;
  }
  if (i.posted_post_id && post(i.posted_post_id)) {
    syncLinkedPost(ideaId);
    return i.posted_post_id;
  }

  const match = db
    .prepare(
      `SELECT sp.id FROM social_posts sp
        WHERE sp.platform = ? AND sp.posted_date = ?
          AND NOT EXISTS (SELECT 1 FROM content_ideas ci WHERE ci.posted_post_id = sp.id AND ci.id != ?)
        ORDER BY sp.id LIMIT 1`,
    )
    .get(i.platform, i.target_date, ideaId) as { id: number } | undefined;

  const postId =
    match?.id ??
    createSocialPost({
      posted_date: i.target_date!,
      platform: i.platform,
      post_type: FORMAT_POST_TYPE[i.format] ?? null,
      caption: postText(i) || null,
      likes: 0,
      comments: 0,
      shares: 0,
      link_clicks: 0,
      notes: PLANNER_LOG_NOTE,
    }).id;
  db.prepare("UPDATE content_ideas SET posted_post_id = ? WHERE id = ?").run(postId, ideaId);
  return postId;
}

/** Undoing "posted": drops the link, and the Post log entry too if the planner
 * made it and nothing real (numbers) has been added to it since. */
export function unlinkPostedIdea(ideaId: number): void {
  const db = getHealthDb();
  const i = idea(ideaId);
  if (!i?.posted_post_id) return;
  const p = post(i.posted_post_id);
  db.prepare("UPDATE content_ideas SET posted_post_id = NULL WHERE id = ?").run(ideaId);
  if (p && plannerCreated(p) && !hasNumbers(p)) db.prepare("DELETE FROM social_posts WHERE id = ?").run(p.id);
}

/** After editing a posted planner item, carry the date, platform, format and
 * text over to the Post log entry the planner created. Entries that came from
 * a Meta import are left alone — they record what actually went out. */
export function syncLinkedPost(ideaId: number): void {
  const i = idea(ideaId);
  if (!i?.posted_post_id || !i.target_date) return;
  const p = post(i.posted_post_id);
  if (!p || !plannerCreated(p)) return;
  getHealthDb()
    .prepare("UPDATE social_posts SET posted_date = ?, platform = ?, post_type = ?, caption = ? WHERE id = ?")
    .run(i.target_date, i.platform, FORMAT_POST_TYPE[i.format] ?? null, postText(i) || null, p.id);
}

/** After posts arrive in the Post log (import or typed in), link any posted
 * planner item still waiting for one on the same platform and day. */
export function linkWaitingPostedIdeas(): number {
  const db = getHealthDb();
  const waiting = db.prepare("SELECT id FROM content_ideas WHERE status = 'used' AND target_date IS NOT NULL AND posted_post_id IS NULL").all() as { id: number }[];
  let linked = 0;
  for (const w of waiting) {
    const i = idea(w.id)!;
    const match = db
      .prepare(
        `SELECT sp.id FROM social_posts sp
          WHERE sp.platform = ? AND sp.posted_date = ?
            AND NOT EXISTS (SELECT 1 FROM content_ideas ci WHERE ci.posted_post_id = sp.id)
          ORDER BY sp.id LIMIT 1`,
      )
      .get(i.platform, i.target_date) as { id: number } | undefined;
    if (match) {
      db.prepare("UPDATE content_ideas SET posted_post_id = ? WHERE id = ?").run(match.id, i.id);
      linked++;
    }
  }
  return linked;
}

/** Post log id → planner item id, for showing "From the planner" in the Post log. */
export function plannerIdeaByPost(): Map<number, number> {
  const rows = getHealthDb().prepare("SELECT id, posted_post_id FROM content_ideas WHERE posted_post_id IS NOT NULL").all() as { id: number; posted_post_id: number }[];
  return new Map(rows.map((r) => [r.posted_post_id, r.id]));
}
