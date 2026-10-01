import { NextRequest, NextResponse } from "next/server";
import { importSocialInsightsCsv } from "@/lib/marketing/socialInsightsImport";
import { linkWaitingPostedIdeas } from "@/lib/social/postLink";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  const body = (await request.json().catch(() => null)) as { csv?: string } | null;
  if (!body || typeof body.csv !== "string" || !body.csv.trim()) {
    return NextResponse.json({ error: "csv (raw CSV text) is required." }, { status: 400 });
  }
  try {
    const result = importSocialInsightsCsv(body.csv);
    linkWaitingPostedIdeas();
    return NextResponse.json({ result });
  } catch (error) {
    console.error("[social-dashboard] failed to import social insights CSV:", error);
    return NextResponse.json({ error: "Unable to import that CSV." }, { status: 500 });
  }
}
