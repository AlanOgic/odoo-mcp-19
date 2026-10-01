import { NextRequest, NextResponse } from "next/server";
import { CalendarItemError, createCalendarItem, listCalendarItems } from "@/lib/social/calendarItems";
import { logActivity } from "@/lib/team/activity";
import { currentPerson, unauthorized } from "@/lib/team/session";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json({ items: listCalendarItems() });
}

export async function POST(request: NextRequest) {
  const person = await currentPerson();
  if (!person) return unauthorized();
  const body = (await request.json().catch(() => null)) as Record<string, unknown> | null;
  try {
    const item = createCalendarItem({ kind: body?.kind, title: body?.title, date: body?.date, notes: body?.notes });
    logActivity(person, "calendar_item", item.id, "create", `Added ${item.kind === "holiday" ? "holiday" : "post-by deadline"} "${item.title}"`);
    return NextResponse.json({ item }, { status: 201 });
  } catch (error) {
    if (error instanceof CalendarItemError) return NextResponse.json({ error: error.message }, { status: 400 });
    console.error("[social-dashboard] failed to add calendar item:", error);
    return NextResponse.json({ error: "Unable to add that to the calendar." }, { status: 500 });
  }
}
