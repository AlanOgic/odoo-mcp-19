import { NextResponse } from "next/server";
import { compareLatestAudits } from "@/lib/issues/auditComparison";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const comparison = compareLatestAudits();
    if (!comparison) return NextResponse.json({ error: "No audits have been imported yet." }, { status: 404 });
    return NextResponse.json({ comparison });
  } catch (error) {
    console.error("[website-health] failed to compare audits:", error);
    return NextResponse.json({ error: "Unable to compare audits." }, { status: 500 });
  }
}
