import "server-only";

import { getHealthDb } from "../db";

/** Holidays and post-by deadlines added by hand on the calendar (the standard
 * ones are computed — see seasonalCalendar.ts). */
export type CalendarItemKind = "holiday" | "deadline";

export interface CalendarItem {
  id: number;
  kind: CalendarItemKind;
  title: string;
  date: string;
  notes: string | null;
  created_at: string;
}

export class CalendarItemError extends Error {}

const DATE = /^\d{4}-\d{2}-\d{2}$/;

function cleanTitle(title: unknown): string {
  const t = typeof title === "string" ? title.trim() : "";
  if (!t) throw new CalendarItemError("Give it a name.");
  if (t.length > 120) throw new CalendarItemError("Keep the name under 120 characters.");
  return t;
}

function cleanDate(date: unknown): string {
  if (typeof date !== "string" || !DATE.test(date)) throw new CalendarItemError("Pick a date.");
  return date;
}

const cleanNotes = (notes: unknown) => (typeof notes === "string" && notes.trim() ? notes.trim() : null);

export function listCalendarItems(): CalendarItem[] {
  return getHealthDb().prepare("SELECT * FROM calendar_items ORDER BY date").all() as CalendarItem[];
}

export function getCalendarItem(id: number): CalendarItem | null {
  return (getHealthDb().prepare("SELECT * FROM calendar_items WHERE id = ?").get(id) as CalendarItem | undefined) ?? null;
}

export function createCalendarItem(data: { kind: unknown; title: unknown; date: unknown; notes?: unknown }): CalendarItem {
  if (data.kind !== "holiday" && data.kind !== "deadline") throw new CalendarItemError('kind must be "holiday" or "deadline".');
  const db = getHealthDb();
  const result = db.prepare("INSERT INTO calendar_items (kind, title, date, notes) VALUES (?, ?, ?, ?)").run(data.kind, cleanTitle(data.title), cleanDate(data.date), cleanNotes(data.notes));
  return getCalendarItem(Number(result.lastInsertRowid))!;
}

export function updateCalendarItem(id: number, data: { title?: unknown; date?: unknown; notes?: unknown }): CalendarItem | null {
  const existing = getCalendarItem(id);
  if (!existing) return null;
  const next = {
    title: data.title !== undefined ? cleanTitle(data.title) : existing.title,
    date: data.date !== undefined ? cleanDate(data.date) : existing.date,
    notes: data.notes !== undefined ? cleanNotes(data.notes) : existing.notes,
  };
  getHealthDb().prepare("UPDATE calendar_items SET title = ?, date = ?, notes = ? WHERE id = ?").run(next.title, next.date, next.notes, id);
  return getCalendarItem(id);
}

export function deleteCalendarItem(id: number): boolean {
  return getHealthDb().prepare("DELETE FROM calendar_items WHERE id = ?").run(id).changes > 0;
}
