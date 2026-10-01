import "server-only";

import { getHealthDb } from "../db";
import type { Person } from "./people";

export type ActivityEntity = "content_idea" | "trade_show" | "social_post" | "trend" | "calendar_item";

export interface ActivityEntry {
  id: number;
  person_id: number | null;
  author_name: string;
  entity_type: ActivityEntity;
  entity_id: number;
  action: string;
  summary: string;
  created_at: string;
}

export function logActivity(person: Pick<Person, "id" | "name">, entity: ActivityEntity, entityId: number, action: string, summary: string): void {
  getHealthDb()
    .prepare("INSERT INTO activity_log (person_id, author_name, entity_type, entity_id, action, summary) VALUES (?, ?, ?, ?, ?, ?)")
    .run(person.id, person.name, entity, entityId, action, summary);
}

/** Newest first, grouped by entity id — one query for a whole list of posts. */
export function activityFor(entity: ActivityEntity, ids?: number[]): Map<number, ActivityEntry[]> {
  const db = getHealthDb();
  const rows = (
    ids && ids.length > 0
      ? db.prepare(`SELECT * FROM activity_log WHERE entity_type = ? AND entity_id IN (${ids.map(() => "?").join(",")}) ORDER BY created_at DESC, id DESC`).all(entity, ...ids)
      : db.prepare("SELECT * FROM activity_log WHERE entity_type = ? ORDER BY created_at DESC, id DESC").all(entity)
  ) as ActivityEntry[];
  const map = new Map<number, ActivityEntry[]>();
  for (const r of rows) (map.get(r.entity_id) ?? map.set(r.entity_id, []).get(r.entity_id)!).push(r);
  return map;
}
