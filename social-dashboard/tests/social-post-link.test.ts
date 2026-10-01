import { describe, it, expect, beforeEach } from "vitest";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

beforeEach(() => {
  process.env.DASHBOARD_DATA_DIR = mkdtempSync(path.join(tmpdir(), "post-link-test-"));
});

async function setup() {
  const ideas = await import("@/lib/social/contentIdeas");
  const link = await import("@/lib/social/postLink");
  const posts = await import("@/lib/marketing/socialPosts");
  const { getHealthDb } = await import("@/lib/db");
  const idea = ideas.createContentIdea({
    idea_type: "new",
    platform: "instagram",
    format: "carousel",
    product: "Shell pendant",
    hook: "Back to the shore",
    caption: "Carve the moment.",
    reasoning: "r",
    confidence: "promising",
  });
  const postedIdea = (id: number) => getHealthDb().prepare("SELECT target_date, posted_post_id FROM content_ideas WHERE id = ?").get(id) as { target_date: string | null; posted_post_id: number | null };
  return { ideas, link, posts, idea, postedIdea, db: getHealthDb() };
}

describe("planner ↔ Post log", () => {
  it("marking posted dates an undated post and creates its Post log entry; un-posting removes it", async () => {
    const { ideas, link, posts, idea, postedIdea } = await setup();
    ideas.updateContentIdeaStatus(idea.id, "used");
    const postId = link.linkPostedIdea(idea.id)!;
    const linked = postedIdea(idea.id);
    expect(linked.target_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(linked.posted_post_id).toBe(postId);
    const entry = posts.listSocialPosts().find((p) => p.id === postId)!;
    expect(entry.caption).toBe("Back to the shore\n\nCarve the moment.");
    expect(entry.post_type).toBe("Carousel");
    expect(entry.notes).toBe(posts.PLANNER_LOG_NOTE);

    ideas.updateContentIdeaStatus(idea.id, "approved");
    link.unlinkPostedIdea(idea.id);
    expect(postedIdea(idea.id).posted_post_id).toBeNull();
    expect(posts.listSocialPosts()).toHaveLength(0);
  });

  it("links to an existing Post log entry for the same platform and day instead of duplicating", async () => {
    const { ideas, link, posts, idea, postedIdea } = await setup();
    const imported = posts.createSocialPost({ posted_date: "2026-09-24", platform: "instagram", caption: "From Meta", likes: 5, comments: 0, shares: 1, link_clicks: 1, notes: "Imported" });
    ideas.updateContentIdea(idea.id, { target_date: "2026-09-24" });
    ideas.updateContentIdeaStatus(idea.id, "used");
    expect(link.linkPostedIdea(idea.id)).toBe(imported.id);
    expect(posts.listSocialPosts()).toHaveLength(1);

    // Un-posting never deletes an entry that has real numbers.
    ideas.updateContentIdeaStatus(idea.id, "approved");
    link.unlinkPostedIdea(idea.id);
    expect(posts.listSocialPosts()).toHaveLength(1);
    expect(postedIdea(idea.id).posted_post_id).toBeNull();
  });

  it("carries edits to the planner's own entry, and a Meta import fills it in rather than duplicating", async () => {
    const { ideas, link, posts, idea } = await setup();
    const { importSocialInsightsCsv } = await import("@/lib/marketing/socialInsightsImport");
    ideas.updateContentIdea(idea.id, { target_date: "2026-09-24" });
    ideas.updateContentIdeaStatus(idea.id, "used");
    const postId = link.linkPostedIdea(idea.id)!;

    ideas.updateContentIdea(idea.id, { target_date: "2026-09-25", caption: "New words." });
    link.syncLinkedPost(idea.id);
    expect(posts.listSocialPosts()[0]!.posted_date).toBe("2026-09-25");

    const csv = [
      "platform,record_type,metric,period_start,period_end,date,value,unit,change_vs_prev_period_pct,post_published,post_title",
      "Instagram,post,Reach,2026-09-01,2026-09-30,2026-09-25,22,count,,2026-09-25 10:00,Take yourself back to the shore.",
      "Instagram,post,Reactions/Likes,2026-09-01,2026-09-30,2026-09-25,4,count,,2026-09-25 10:00,Take yourself back to the shore.",
    ].join("\n");
    importSocialInsightsCsv(csv);
    const all = posts.listSocialPosts();
    expect(all).toHaveLength(1);
    expect(all[0]!.id).toBe(postId);
    expect(all[0]!.likes).toBe(4);
  });

  it("links a posted item that was waiting once its post arrives in the Post log", async () => {
    const { ideas, link, posts, idea, postedIdea, db } = await setup();
    ideas.updateContentIdea(idea.id, { target_date: "2026-09-24" });
    ideas.updateContentIdeaStatus(idea.id, "used");
    const p = posts.createSocialPost({ posted_date: "2026-09-24", platform: "instagram", caption: "typed in", likes: 0, comments: 0, shares: 0, link_clicks: 0 });
    expect(link.linkWaitingPostedIdeas()).toBe(1);
    expect(postedIdea(idea.id).posted_post_id).toBe(p.id);
    expect(link.plannerIdeaByPost().get(p.id)).toBe(idea.id);
    db.prepare("DELETE FROM social_posts WHERE id = ?").run(p.id);
    expect(postedIdea(idea.id).posted_post_id).toBeNull();
  });
});
