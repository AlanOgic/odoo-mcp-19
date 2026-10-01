import { NextRequest, NextResponse } from "next/server";
import { listContentIdeas, createContentIdea, updateContentIdeaStatus, type NewContentIdea } from "@/lib/social/contentIdeas";
import { linkPostedIdea } from "@/lib/social/postLink";
import { commentsFor } from "@/lib/team/comments";
import { activityFor, logActivity } from "@/lib/team/activity";
import { currentPerson, unauthorized } from "@/lib/team/session";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const comments = commentsFor();
    const activity = activityFor("content_idea");
    const ideas = listContentIdeas().map((idea) => ({ ...idea, comments: comments.get(idea.id) ?? [], activity: activity.get(idea.id) ?? [] }));
    return NextResponse.json({ ideas });
  } catch (error) {
    console.error("[social-dashboard] failed to list content ideas:", error);
    return NextResponse.json({ error: "Unable to load content ideas." }, { status: 500 });
  }
}

const PLATFORMS = ["instagram", "facebook", "tiktok", "pinterest"];
const FORMATS = ["photo", "reel", "carousel", "story"];

/** A post someone adds straight onto the calendar — scheduled, or already posted.
 * Only a title, date, platform and format are needed; the rest can be filled in later. */
function createManualPost(body: Record<string, unknown>, personName: string) {
  const title = typeof body.product === "string" ? body.product.trim() : "";
  if (!title) return { error: "Give the post a title." };
  if (typeof body.target_date !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(body.target_date)) return { error: "Pick a date." };
  if (typeof body.platform !== "string" || !PLATFORMS.includes(body.platform)) return { error: "Pick a platform." };
  const format = typeof body.format === "string" && FORMATS.includes(body.format) ? body.format : "photo";
  const posted = body.status === "used";
  const idea = createContentIdea({
    idea_type: "new",
    platform: body.platform as NewContentIdea["platform"],
    format: format as NewContentIdea["format"],
    product: title,
    target_date: body.target_date,
    caption: typeof body.caption === "string" ? body.caption.trim() : "",
    reasoning: `Added on the calendar by ${personName}.`,
    confidence: "insufficient",
  });
  updateContentIdeaStatus(idea.id, posted ? "used" : "approved");
  if (posted) linkPostedIdea(idea.id);
  return { idea, posted };
}

export async function POST(request: NextRequest) {
  const person = await currentPerson();
  if (!person) return unauthorized();
  const raw = (await request.json().catch(() => null)) as Record<string, unknown> | null;
  if (raw?.manual) {
    const result = createManualPost(raw, person.name);
    if ("error" in result) return NextResponse.json({ error: result.error }, { status: 400 });
    logActivity(person, "content_idea", result.idea.id, "create", result.posted ? "Logged as posted from the calendar" : "Scheduled from the calendar");
    return NextResponse.json({ idea: result.idea }, { status: 201 });
  }
  const body = raw as Partial<NewContentIdea> | null;
  if (!body || !body.idea_type || !body.platform || !body.product || !body.caption || !body.reasoning || !body.confidence) {
    return NextResponse.json({ error: "idea_type, platform, product, caption, reasoning, and confidence are required." }, { status: 400 });
  }
  try {
    const idea = createContentIdea(body as NewContentIdea);
    logActivity(person, "content_idea", idea.id, "create", "Created this post");
    return NextResponse.json({ idea }, { status: 201 });
  } catch (error) {
    console.error("[social-dashboard] failed to create content idea:", error);
    return NextResponse.json({ error: "Unable to save content idea." }, { status: 500 });
  }
}
