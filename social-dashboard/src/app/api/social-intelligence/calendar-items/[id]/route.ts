import { NextRequest, NextResponse } from "next/server";
import { CalendarItemError, deleteCalendarItem, getCalendarItem, updateCalendarItem } from "@/lib/social/calendarItems";
import { logActivity } from "@/lib/team/activity";
import { currentPerson, unauthorized } from "@/lib/team/session";

export const dynamic = "force-dynamic";

export async function PATCH(request: NextRequest, context: { params: Promise<{ id: string }> }) {
  const person = await currentPerson();
  if (!person) return unauthorized();
  const { id } = await context.params;
  const before = getCalendarItem(Number(id));
  if (!before) return NextResponse.json({ error: "That calendar item no longer exists." }, { status: 404 });
  const body = (await request.json().catch(() => ({}))) as Record<string, unknown>;
  try {
    const item = updateCalendarItem(Number(id), { title: body.title, date: body.date, notes: body.notes })!;
    if (item.date !== before.date) logActivity(person, "calendar_item", item.id, "edit", `Date ${before.date} → ${item.date}`);
    if (item.title !== before.title) logActivity(person, "calendar_item", item.id, "edit", "Renamed");
    if (item.notes !== before.notes) logActivity(person, "calendar_item", item.id, "note", item.notes ? "Edited the note" : "Removed the note");
    return NextResponse.json({ item });
  } catch (error) {
    if (error instanceof CalendarItemError) return NextResponse.json({ error: error.message }, { status: 400 });
    throw error;
  }
}

export async function DELETE(_request: NextRequest, context: { params: Promise<{ id: string }> }) {
  const person = await currentPerson();
  if (!person) return unauthorized();
  const { id } = await context.params;
  const before = getCalendarItem(Number(id));
  if (!before || !deleteCalendarItem(Number(id))) return NextResponse.json({ error: "That calendar item no longer exists." }, { status: 404 });
  logActivity(person, "calendar_item", Number(id), "delete", `Deleted "${before.title}"`);
  return NextResponse.json({ ok: true });
}
