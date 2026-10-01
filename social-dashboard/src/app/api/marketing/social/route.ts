import { NextRequest, NextResponse } from "next/server";
import { listSocialPosts, createSocialPost, type NewSocialPost, type SocialPlatform } from "@/lib/marketing/socialPosts";
import { linkWaitingPostedIdeas, plannerIdeaByPost } from "@/lib/social/postLink";

export const dynamic = "force-dynamic";

const VALID_PLATFORMS: SocialPlatform[] = ["instagram", "facebook", "tiktok", "pinterest"];

export async function GET(request: NextRequest) {
  const platform = request.nextUrl.searchParams.get("platform") as SocialPlatform | null;
  try {
    const planned = plannerIdeaByPost();
    const posts = listSocialPosts(platform && VALID_PLATFORMS.includes(platform) ? platform : undefined).map((p) => ({ ...p, planner_idea_id: planned.get(p.id) ?? null }));
    return NextResponse.json({ posts });
  } catch (error) {
    console.error("[social-dashboard] failed to list social posts:", error);
    return NextResponse.json({ error: "Unable to load social posts." }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  const body = (await request.json().catch(() => null)) as Partial<NewSocialPost> | null;
  if (!body || !body.posted_date || !body.platform || !VALID_PLATFORMS.includes(body.platform)) {
    return NextResponse.json({ error: "posted_date and a valid platform are required." }, { status: 400 });
  }
  try {
    const post = createSocialPost({
      posted_date: body.posted_date,
      platform: body.platform,
      post_type: body.post_type ?? null,
      caption: body.caption ?? null,
      reach: body.reach != null ? Number(body.reach) : null,
      likes: Number(body.likes) || 0,
      comments: Number(body.comments) || 0,
      shares: Number(body.shares) || 0,
      link_clicks: Number(body.link_clicks) || 0,
      notes: body.notes ?? null,
      utm_source: body.utm_source ?? null,
      utm_medium: body.utm_medium ?? null,
      utm_campaign: body.utm_campaign ?? null,
      revenue_attributed: Number(body.revenue_attributed) || 0,
    });
    linkWaitingPostedIdeas();
    return NextResponse.json({ post }, { status: 201 });
  } catch (error) {
    console.error("[social-dashboard] failed to create social post:", error);
    return NextResponse.json({ error: "Unable to save social post." }, { status: 500 });
  }
}
