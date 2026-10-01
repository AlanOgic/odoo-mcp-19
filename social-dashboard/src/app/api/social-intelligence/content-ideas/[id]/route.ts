import { NextRequest, NextResponse } from "next/server";
import {
  updateContentIdeaStatus,
  updateContentIdea,
  recomputeSuggestedTime,
  deleteContentIdea,
  listContentIdeas,
  type ContentIdeaStatus,
  type ContentIdeaUpdate,
  type ContentIdea,
} from "@/lib/social/contentIdeas";
import { describeIdeaEdit, parseHashtagInput, type EditableIdeaFields } from "@/lib/social/contentIdeaEdits";
import { logActivity } from "@/lib/team/activity";
import { linkPostedIdea, syncLinkedPost, unlinkPostedIdea } from "@/lib/social/postLink";
import { currentPerson, unauthorized } from "@/lib/team/session";

export const dynamic = "force-dynamic";

const VALID_STATUSES: ContentIdeaStatus[] = ["suggested", "approved", "used", "dismissed"];
const VALID_PLATFORMS = ["instagram", "facebook", "tiktok", "pinterest"];
const VALID_FORMATS = ["photo", "reel", "carousel", "story"];
const STATUS_SUMMARY: Record<ContentIdeaStatus, string> = {
  suggested: "Moved back to suggested",
  approved: "Approved",
  used: "Marked as posted",
  dismissed: "Dismissed",
};

type PatchBody = { status?: string; recompute_time?: boolean; hashtags_text?: string } & ContentIdeaUpdate;

function findIdea(id: number): ContentIdea | undefined {
  return listContentIdeas().find((i) => i.id === id);
}

function editableFields(idea: ContentIdea): EditableIdeaFields {
  return {
    product: idea.product,
    hook: idea.hook,
    caption: idea.caption,
    hashtags: idea.hashtags,
    cta: idea.cta,
    target_date: idea.target_date,
    platform: idea.platform,
    format: idea.format,
    pillar: idea.pillar,
  };
}

const optionalText = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : null);

/** Builds a validated update from the request, or an error message for the person editing. */
function buildUpdate(body: PatchBody, existing: ContentIdea): { update: ContentIdeaUpdate } | { error: string } {
  const update: ContentIdeaUpdate = {};
  if (body.product !== undefined) {
    if (typeof body.product !== "string" || !body.product.trim()) return { error: "The post needs a title." };
    update.product = body.product.trim();
  }
  if (body.caption !== undefined) update.caption = typeof body.caption === "string" ? body.caption.trim() : "";
  if (body.hook !== undefined) update.hook = optionalText(body.hook);
  if (body.cta !== undefined) update.cta = optionalText(body.cta);
  if (body.pillar !== undefined) update.pillar = optionalText(body.pillar);
  if (body.hashtags_text !== undefined) update.hashtags = parseHashtagInput(String(body.hashtags_text), existing.hashtags);
  else if (body.hashtags !== undefined) update.hashtags = body.hashtags;
  if (body.target_date !== undefined) {
    if (body.target_date !== null && !/^\d{4}-\d{2}-\d{2}$/.test(String(body.target_date))) return { error: "Pick a valid date, or clear it to unschedule." };
    update.target_date = body.target_date || null;
  }
  if (body.platform !== undefined) {
    if (!VALID_PLATFORMS.includes(body.platform)) return { error: `Platform must be one of: ${VALID_PLATFORMS.join(", ")}.` };
    update.platform = body.platform;
  }
  if (body.format !== undefined) {
    if (!VALID_FORMATS.includes(body.format)) return { error: `Format must be one of: ${VALID_FORMATS.join(", ")}.` };
    update.format = body.format;
  }
  if (body.suggested_time !== undefined) update.suggested_time = body.suggested_time;
  return { update };
}

export async function PATCH(request: NextRequest, context: { params: Promise<{ id: string }> }) {
  const person = await currentPerson();
  if (!person) return unauthorized();
  const { id } = await context.params;
  const ideaId = Number(id);
  const body = (await request.json().catch(() => ({}))) as PatchBody;

  const existing = findIdea(ideaId);
  if (!existing) return NextResponse.json({ error: "Idea not found." }, { status: 404 });

  if (body.recompute_time) {
    const idea = recomputeSuggestedTime(ideaId);
    if (!idea) return NextResponse.json({ error: "Idea not found or has no target_date." }, { status: 404 });
    return NextResponse.json({ idea });
  }

  if (body.status !== undefined) {
    if (!VALID_STATUSES.includes(body.status as ContentIdeaStatus)) {
      return NextResponse.json({ error: `status must be one of: ${VALID_STATUSES.join(", ")}.` }, { status: 400 });
    }
    const updated = updateContentIdeaStatus(ideaId, body.status as ContentIdeaStatus);
    if (!updated) return NextResponse.json({ error: "Idea not found." }, { status: 404 });
    // Keep the Post log in step: posted items get an entry there, un-posting removes an empty one.
    if (updated.status === "used") linkPostedIdea(ideaId);
    else if (existing.status === "used") unlinkPostedIdea(ideaId);
    if (existing.status !== updated.status) logActivity(person, "content_idea", ideaId, "status", STATUS_SUMMARY[updated.status]);
    return NextResponse.json({ idea: findIdea(ideaId) });
  }

  const built = buildUpdate(body, existing);
  if ("error" in built) return NextResponse.json({ error: built.error }, { status: 400 });
  const summary = describeIdeaEdit(editableFields(existing), built.update as Partial<EditableIdeaFields>);
  if (!summary) return NextResponse.json({ idea: existing });

  const idea = updateContentIdea(ideaId, built.update);
  if (!idea) return NextResponse.json({ error: "Idea not found." }, { status: 404 });
  if (idea.status === "used") syncLinkedPost(ideaId);
  logActivity(person, "content_idea", ideaId, "edit", summary);
  return NextResponse.json({ idea });
}

export async function DELETE(_request: NextRequest, context: { params: Promise<{ id: string }> }) {
  const person = await currentPerson();
  if (!person) return unauthorized();
  const { id } = await context.params;
  const existing = findIdea(Number(id));
  const deleted = deleteContentIdea(Number(id));
  if (!deleted) return NextResponse.json({ error: "Idea not found." }, { status: 404 });
  logActivity(person, "content_idea", Number(id), "delete", `Deleted "${existing?.product ?? "post"}"`);
  return NextResponse.json({ ok: true });
}
