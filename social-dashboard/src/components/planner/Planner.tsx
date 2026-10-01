"use client";

import { useEffect, useMemo, useState } from "react";
import { MonthCalendar, type CalendarDayEntry } from "@/components/ui/MonthCalendar";
import {
  buildCalendarEntries,
  upcomingDeadlineAlerts,
  type CalendarCustomItem,
  type CalendarEntry,
  type CalendarOccasion,
  type CalendarTradeShow,
} from "@/lib/social/calendarEntries";
import { PostDrawer } from "./PostDrawer";
import { FORMAT_LABEL, PLATFORM_LABEL, STATUS_LABEL, postStats, readError, shortDate, statusPillClass, type Me, type PlannerIdea, type PlannerLoggedPost } from "./types";

/**
 * Colour rule for the calendar, kept to the board's two highlights:
 * teal = needs doing (scheduled posts, post-by deadlines), lavender = settled
 * (posted, events you're attending), grey = context (holidays).
 */
type DayKind = "scheduled" | "deadline" | "posted" | "event" | "holiday";
const DAY_KIND: Record<DayKind, { label: string; color: string }> = {
  scheduled: { label: "Scheduled post", color: "var(--accent)" },
  deadline: { label: "Post-by deadline", color: "var(--accent)" },
  posted: { label: "Posted", color: "var(--second)" },
  event: { label: "Event", color: "var(--second)" },
  holiday: { label: "Holiday", color: "var(--text-faint)" },
};

const UPCOMING_DAYS = 14;

function todayIso(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function addDays(iso: string, days: number): string {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function daysBetween(from: string, to: string): number {
  return Math.round((new Date(`${to}T00:00:00Z`).getTime() - new Date(`${from}T00:00:00Z`).getTime()) / 86_400_000);
}

function without<T>(record: Record<number, T>, id: number): Record<number, T> {
  const next = { ...record };
  delete next[id];
  return next;
}

interface ShowDates {
  start: string;
  end: string | null;
}

interface Toast {
  message: string;
  error?: boolean;
  undo?: () => void;
}

export function Planner({
  ideas,
  occasions,
  tradeShows,
  customItems,
  loggedPosts,
  loading,
  me,
  onIdeasChanged,
  onCalendarChanged,
}: {
  ideas: PlannerIdea[];
  occasions: CalendarOccasion[];
  tradeShows: CalendarTradeShow[];
  customItems: CalendarCustomItem[];
  loggedPosts: PlannerLoggedPost[];
  loading: boolean;
  me: Me | null;
  onIdeasChanged: () => void | Promise<void>;
  onCalendarChanged: () => void | Promise<void>;
}) {
  const [openIdeaId, setOpenIdeaId] = useState<number | null>(null);
  // Drag-and-drop moves show in their new place at once, before the save round-trip finishes.
  const [ideaMoves, setIdeaMoves] = useState<Record<number, string>>({});
  const [showMoves, setShowMoves] = useState<Record<number, ShowDates>>({});
  const [customMoves, setCustomMoves] = useState<Record<number, string>>({});
  const [toast, setToast] = useState<Toast | null>(null);

  // The Post log links here with ?idea=<id> to open that post.
  useEffect(() => {
    const id = Number(new URLSearchParams(window.location.search).get("idea"));
    if (id) {
      setOpenIdeaId(id);
      window.history.replaceState(null, "", window.location.pathname);
    }
  }, []);

  useEffect(() => {
    if (!toast) return;
    const t = window.setTimeout(() => setToast(null), toast.undo ? 7000 : 5000);
    return () => window.clearTimeout(t);
  }, [toast]);

  const shownIdeas = useMemo(() => ideas.map((i) => (ideaMoves[i.id] !== undefined ? { ...i, target_date: ideaMoves[i.id]! } : i)), [ideas, ideaMoves]);
  const shownShows = useMemo(
    () =>
      tradeShows.map((t) => {
        const m = showMoves[t.id];
        return m ? { ...t, start_date: m.start, end_date: m.end, post_by_date: addDays(m.start, -t.lead_days) } : t;
      }),
    [tradeShows, showMoves],
  );

  const shownCustom = useMemo(() => customItems.map((c) => (customMoves[c.id] !== undefined ? { ...c, date: customMoves[c.id]! } : c)), [customItems, customMoves]);

  const openIdea = shownIdeas.find((i) => i.id === openIdeaId) ?? null;
  const ideaById = useMemo(() => new Map(shownIdeas.map((i) => [i.id, i])), [shownIdeas]);
  const loggedPostById = useMemo(() => new Map(loggedPosts.map((p) => [p.id, p])), [loggedPosts]);

  const entries = useMemo(
    () =>
      buildCalendarEntries(
        occasions,
        shownShows,
        shownIdeas.map((i) => ({
          id: i.id,
          idea_type: i.idea_type,
          target_date: i.target_date,
          product: i.product,
          platform: i.platform,
          status: i.status,
          format: i.format,
          posted_post_id: i.posted_post_id,
        })),
        shownCustom,
        loggedPosts,
      ),
    [occasions, shownShows, shownIdeas, shownCustom, loggedPosts],
  );

  const kindOf = (e: CalendarEntry): DayKind => {
    if (e.loggedPostId != null) return "posted";
    if (e.ideaId != null) return ideaById.get(e.ideaId)?.status === "used" ? "posted" : "scheduled";
    if (e.type === "post-deadline") return "deadline";
    if (e.type === "trade-show") return "event";
    return "holiday";
  };

  const deadlines = useMemo(() => upcomingDeadlineAlerts(entries, 2), [entries]);

  async function moveIdea(id: number, name: string, fromDate: string | null, toDate: string, offerUndo: boolean) {
    setIdeaMoves((m) => ({ ...m, [id]: toDate }));
    const res = await fetch(`/api/social-intelligence/content-ideas/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_date: toDate }),
    }).catch(() => null);
    if (res?.ok) await onIdeasChanged();
    setIdeaMoves((m) => without(m, id));
    if (!res?.ok) return setToast({ message: await readError(res, `Couldn't move "${name}".`), error: true });
    setToast({
      message: `Moved "${name}" to ${shortDate(toDate)}`,
      undo: offerUndo && fromDate ? () => moveIdea(id, name, toDate, fromDate, false) : undefined,
    });
  }

  async function moveShow(id: number, name: string, from: ShowDates, to: ShowDates, offerUndo: boolean) {
    setShowMoves((m) => ({ ...m, [id]: to }));
    const res = await fetch(`/api/social-intelligence/trade-shows/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start_date: to.start, end_date: to.end }),
    }).catch(() => null);
    if (res?.ok) await onCalendarChanged();
    setShowMoves((m) => without(m, id));
    if (!res?.ok) return setToast({ message: await readError(res, `Couldn't move "${name}".`), error: true });
    setToast({
      message: `Moved "${name}" to ${shortDate(to.start)}${to.end && to.end !== to.start ? ` – ${shortDate(to.end)}` : ""}`,
      undo: offerUndo ? () => moveShow(id, name, to, from, false) : undefined,
    });
  }

  async function moveCustom(id: number, name: string, fromDate: string, toDate: string, offerUndo: boolean) {
    setCustomMoves((m) => ({ ...m, [id]: toDate }));
    const res = await fetch(`/api/social-intelligence/calendar-items/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date: toDate }),
    }).catch(() => null);
    if (res?.ok) await onCalendarChanged();
    setCustomMoves((m) => without(m, id));
    if (!res?.ok) return setToast({ message: await readError(res, `Couldn't move "${name}".`), error: true });
    setToast({ message: `Moved "${name}" to ${shortDate(toDate)}`, undo: offerUndo ? () => moveCustom(id, name, toDate, fromDate, false) : undefined });
  }

  /** Posts still to go out, events, and holidays/deadlines the team added can be
   * rescheduled by dragging; computed holidays and deadlines, and anything already
   * posted (history), can't. */
  const canDrag = (entryKey: string): boolean => {
    const e = entries.find((x) => x.key === entryKey);
    if (!e) return false;
    if (e.customItemId != null) return true;
    const kind = kindOf(e);
    return kind === "scheduled" || kind === "event";
  };

  function handleMove(entryKey: string, fromDate: string, toDate: string) {
    const e = entries.find((x) => x.key === entryKey);
    if (!e) return;
    if (e.ideaId != null) {
      const original = ideas.find((i) => i.id === e.ideaId);
      if (original) moveIdea(original.id, original.product, original.target_date, toDate, true);
      return;
    }
    if (e.customItemId != null) {
      const item = customItems.find((c) => c.id === e.customItemId);
      if (item) moveCustom(item.id, item.title, item.date, toDate, true);
      return;
    }
    if (e.deletableTradeShowId != null) {
      const show = tradeShows.find((t) => t.id === e.deletableTradeShowId);
      if (!show) return;
      const delta = daysBetween(fromDate, toDate);
      const from = { start: show.start_date, end: show.end_date };
      const to = { start: addDays(show.start_date, delta), end: show.end_date ? addDays(show.end_date, delta) : null };
      moveShow(show.id, show.name, from, to, true);
    }
  }

  const today = todayIso();
  const horizon = addDays(today, UPCOMING_DAYS);
  const upcoming = shownIdeas
    .filter((i) => i.target_date && i.target_date >= today && i.target_date <= horizon && i.status !== "dismissed")
    .sort((a, b) => (a.target_date! + (a.suggested_time ?? "")).localeCompare(b.target_date! + (b.suggested_time ?? "")));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {deadlines.length > 0 && (
        <div className="card" style={{ padding: "14px 18px", borderColor: "var(--accent-border)", display: "flex", gap: 14, alignItems: "flex-start" }}>
          <span className="pill pill-accent" style={{ flexShrink: 0 }}>
            <span className="dot" />
            Due soon
          </span>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {deadlines.map((d) => (
              <p key={d.key} style={{ margin: 0, fontSize: 13 }}>
                <strong style={{ fontWeight: 600 }}>{shortDate(d.date)}</strong>
                <span className="muted"> · {d.label.replace(/^Post-by deadline: /, "Have a post live for ")}</span>
              </p>
            ))}
          </div>
        </div>
      )}

      <div className="planner-grid">
        <UpcomingList ideas={upcoming} loading={loading} onOpen={setOpenIdeaId} />
        <CalendarCard
          entries={entries}
          kindOf={kindOf}
          ideaById={ideaById}
          loggedPostById={loggedPostById}
          onOpenIdea={setOpenIdeaId}
          onChanged={onCalendarChanged}
          canDrag={canDrag}
          onMoveEntry={handleMove}
        />
      </div>

      <IdeasCard ideas={shownIdeas} loading={loading} onOpen={setOpenIdeaId} />

      {openIdea && (
        <PostDrawer
          idea={openIdea}
          loggedPost={openIdea.posted_post_id != null ? loggedPostById.get(openIdea.posted_post_id) : undefined}
          me={me}
          onClose={() => setOpenIdeaId(null)}
          onChanged={() => void onIdeasChanged()}
        />
      )}

      {toast && (
        <div
          role="status"
          className="card"
          style={{
            position: "fixed",
            left: "50%",
            bottom: "calc(24px + env(safe-area-inset-bottom, 0px))",
            transform: "translateX(-50%)",
            zIndex: 150,
            padding: "10px 14px",
            display: "flex",
            alignItems: "center",
            gap: 14,
            maxWidth: "calc(100% - 32px)",
            background: "var(--surface-raised)",
            borderColor: toast.error ? "var(--negative)" : "var(--border-strong)",
            boxShadow: "0 10px 30px rgba(0, 0, 0, 0.5)",
          }}
        >
          <span style={{ fontSize: 13, color: toast.error ? "var(--negative)" : "var(--text)" }}>{toast.message}</span>
          {toast.undo && (
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => {
                const undo = toast.undo!;
                setToast(null);
                undo();
              }}
            >
              Undo
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function PostRow({ idea, onOpen, showDate = true }: { idea: PlannerIdea; onOpen: (id: number) => void; showDate?: boolean }) {
  return (
    <button
      type="button"
      onClick={() => onOpen(idea.id)}
      style={{
        display: "grid",
        gridTemplateColumns: showDate ? "52px minmax(0, 1fr) auto" : "minmax(0, 1fr) auto",
        gap: 12,
        alignItems: "center",
        width: "100%",
        textAlign: "left",
        background: "transparent",
        border: "none",
        borderBottom: "1px solid var(--border)",
        padding: "12px 0",
        cursor: "pointer",
      }}
    >
      {showDate && (
        <span className="tabular" style={{ lineHeight: 1.25 }}>
          {idea.target_date ? (
            <>
              <span style={{ display: "block", fontSize: 11, color: "var(--text-faint)" }}>{new Date(`${idea.target_date}T00:00:00`).toLocaleDateString("en-US", { weekday: "short" })}</span>
              <span style={{ display: "block", fontSize: 13, color: "var(--text)" }}>{new Date(`${idea.target_date}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" })}</span>
            </>
          ) : (
            "—"
          )}
        </span>
      )}
      <span style={{ minWidth: 0 }}>
        <span style={{ display: "block", fontSize: 13.5, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{idea.product}</span>
        <span style={{ display: "block", fontSize: 12, color: "var(--text-faint)", marginTop: 2 }}>
          {PLATFORM_LABEL[idea.platform] ?? idea.platform} · {FORMAT_LABEL[idea.format]}
          {idea.suggested_time && idea.target_date ? ` · ${idea.suggested_time}` : ""}
          {idea.comments.length > 0 ? ` · ${idea.comments.length} comment${idea.comments.length === 1 ? "" : "s"}` : ""}
        </span>
      </span>
      <span className={statusPillClass(idea.status)}>
        <span className="dot" />
        {STATUS_LABEL[idea.status]}
      </span>
    </button>
  );
}

function UpcomingList({ ideas, loading, onOpen }: { ideas: PlannerIdea[]; loading: boolean; onOpen: (id: number) => void }) {
  return (
    <section className="card" style={{ padding: "20px 22px" }}>
      <h2 className="font-display" style={{ fontSize: 16, margin: "0 0 4px" }}>
        Next two weeks
      </h2>
      <p className="muted" style={{ fontSize: 13, margin: "0 0 8px" }}>
        {loading ? "Loading…" : ideas.length === 0 ? "Nothing scheduled yet." : `${ideas.length} post${ideas.length === 1 ? "" : "s"} going out`}
      </p>
      <div>
        {ideas.map((i) => (
          <PostRow key={i.id} idea={i} onOpen={onOpen} />
        ))}
      </div>
    </section>
  );
}

function CalendarCard({
  entries,
  kindOf,
  ideaById,
  loggedPostById,
  onOpenIdea,
  onChanged,
  canDrag,
  onMoveEntry,
}: {
  entries: CalendarEntry[];
  kindOf: (e: CalendarEntry) => DayKind;
  ideaById: Map<number, PlannerIdea>;
  loggedPostById: Map<number, PlannerLoggedPost>;
  onOpenIdea: (id: number) => void;
  onChanged: () => void | Promise<void>;
  canDrag: (entryKey: string) => boolean;
  onMoveEntry: (entryKey: string, fromDate: string, toDate: string) => void;
}) {
  const now = new Date();
  const [viewYear, setViewYear] = useState(now.getFullYear());
  const [viewMonth, setViewMonth] = useState(now.getMonth());
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  // The Add form is open when this is set; its value is the date to start from.
  const [addDate, setAddDate] = useState<string | null>(null);

  const byDate = useMemo(() => {
    const map: Record<string, CalendarEntry[]> = {};
    for (const e of entries) (map[e.date] ??= []).push(e);
    return map;
  }, [entries]);

  const gridEntries = useMemo(() => {
    const map: Record<string, CalendarDayEntry[]> = {};
    for (const [date, list] of Object.entries(byDate)) {
      map[date] = list.map((e) => {
        const idea = e.ideaId != null ? ideaById.get(e.ideaId) : undefined;
        const label = idea ? `${idea.format === "story" ? "Story · " : ""}${idea.product}` : e.type === "post-deadline" ? e.label.replace(/^Post-by deadline: /, "Post by · ") : e.label;
        return { id: e.key, label, kind: kindOf(e) };
      });
    }
    return map;
  }, [byDate, ideaById, kindOf]);

  const shift = (delta: number) => {
    const d = new Date(Date.UTC(viewYear, viewMonth + delta, 1));
    setViewYear(d.getUTCFullYear());
    setViewMonth(d.getUTCMonth());
  };

  const selected = selectedDate ? (byDate[selectedDate] ?? []) : [];

  return (
    <section className="card" style={{ padding: "20px 22px", minWidth: 0 }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 12, marginBottom: 14, flexWrap: "wrap" }}>
        <div style={{ marginRight: "auto" }}>
          <h2 className="font-display" style={{ fontSize: 16, margin: "0 0 8px" }}>
            Calendar
          </h2>
          <p style={{ margin: "0 0 8px", fontSize: 12, color: "var(--text-faint)" }}>Drag a post or event to another day to reschedule it. On a phone, press and hold first.</p>
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
            {(Object.keys(DAY_KIND) as DayKind[]).map((k) => (
              <span key={k} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-soft)" }}>
                <span aria-hidden style={{ width: 6, height: 6, borderRadius: "50%", background: DAY_KIND[k].color }} />
                {DAY_KIND[k].label}
              </span>
            ))}
          </div>
        </div>
        <button type="button" className={`btn btn-sm ${addDate !== null ? "btn-ghost" : ""}`} onClick={() => setAddDate((d) => (d !== null ? null : ""))}>
          {addDate !== null ? "Cancel" : "Add to calendar"}
        </button>
      </div>

      {addDate !== null && (
        <AddToCalendarForm
          key={addDate}
          initialDate={addDate}
          onCancel={() => setAddDate(null)}
          onSaved={async () => {
            setAddDate(null);
            await onChanged();
          }}
        />
      )}

      <MonthCalendar
        year={viewYear}
        month={viewMonth}
        entriesByDate={gridEntries}
        selectedDate={selectedDate}
        onPrevMonth={() => shift(-1)}
        onNextMonth={() => shift(1)}
        onToday={() => {
          setViewYear(now.getFullYear());
          setViewMonth(now.getMonth());
        }}
        onSelectDay={(iso) => setSelectedDate(iso === selectedDate ? null : iso)}
        kindColor={(kind) => DAY_KIND[kind as DayKind]?.color ?? "var(--text-faint)"}
        canDrag={canDrag}
        onMoveEntry={onMoveEntry}
      />

      <details className="about" style={{ marginTop: 14 }}>
        <summary>How to read the calendar</summary>
        <p>
          A post-by deadline is the last day to have something live for an upcoming holiday or event. It sits before the date itself, and
          any day up to it works. Holidays are computed from the calendar, never guessed. Posting times are general research, not measured
          from this account.
        </p>
      </details>

      {selectedDate && (
        <div className="overlay" style={{ alignItems: "center", justifyContent: "center" }} onClick={() => setSelectedDate(null)}>
          <div className="modal" role="dialog" aria-modal="true" aria-label={shortDate(selectedDate)} onClick={(e) => e.stopPropagation()}>
            <div style={{ display: "flex", alignItems: "center", marginBottom: 16 }}>
              <strong className="font-display" style={{ fontSize: 16, marginRight: "auto" }}>
                {shortDate(selectedDate)}
              </strong>
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => {
                  setAddDate(selectedDate);
                  setSelectedDate(null);
                }}
              >
                Add to this day
              </button>
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => setSelectedDate(null)} aria-label="Close">
                ✕
              </button>
            </div>
            {selected.length === 0 ? (
              <p className="muted" style={{ margin: 0 }}>
                Nothing on this day.
              </p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {selected.map((e) => (
                  <DayEntry
                    key={e.key}
                    entry={e}
                    kind={kindOf(e)}
                    idea={e.ideaId != null ? ideaById.get(e.ideaId) : undefined}
                    loggedPost={
                      e.loggedPostId != null
                        ? loggedPostById.get(e.loggedPostId)
                        : e.ideaId != null && ideaById.get(e.ideaId)?.posted_post_id != null
                          ? loggedPostById.get(ideaById.get(e.ideaId)!.posted_post_id!)
                          : undefined
                    }
                    onOpenIdea={(id) => {
                      setSelectedDate(null);
                      onOpenIdea(id);
                    }}
                    onChanged={onChanged}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

function DayEntry({
  entry,
  kind,
  idea,
  loggedPost,
  onOpenIdea,
  onChanged,
}: {
  entry: CalendarEntry;
  kind: DayKind;
  idea?: PlannerIdea;
  loggedPost?: PlannerLoggedPost;
  onOpenIdea: (id: number) => void;
  onChanged: () => void | Promise<void>;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const deleteTarget =
    entry.deletableTradeShowId != null
      ? { url: `/api/social-intelligence/trade-shows/${entry.deletableTradeShowId}`, what: "event" }
      : entry.customItemId != null
        ? { url: `/api/social-intelligence/calendar-items/${entry.customItemId}`, what: entry.type === "occasion" ? "holiday" : "deadline" }
        : null;

  async function remove() {
    if (!deleteTarget) return;
    await fetch(deleteTarget.url, { method: "DELETE" });
    await onChanged();
  }

  const stats = loggedPost ? postStats(loggedPost) : null;

  return (
    <div style={{ display: "flex", gap: 10 }}>
      <span aria-hidden style={{ width: 6, height: 6, borderRadius: "50%", background: DAY_KIND[kind].color, marginTop: 7, flexShrink: 0 }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 11.5, color: "var(--text-faint)", marginBottom: 2 }}>{DAY_KIND[kind].label}</div>
        <div style={{ fontSize: 14, fontWeight: 500 }}>{idea ? idea.product : entry.label.replace(/^Post-by deadline: /, "")}</div>
        {idea ? (
          <>
            <p className="muted" style={{ margin: "2px 0 8px", fontSize: 12.5 }}>
              {PLATFORM_LABEL[idea.platform] ?? idea.platform} · {FORMAT_LABEL[idea.format]} · {STATUS_LABEL[idea.status]}
              {idea.comments.length > 0 ? ` · ${idea.comments.length} comment${idea.comments.length === 1 ? "" : "s"}` : ""}
            </p>
            {loggedPost && <PostLogLine stats={stats} />}
            <button type="button" className="btn btn-sm" onClick={() => onOpenIdea(idea.id)}>
              Open post
            </button>
          </>
        ) : loggedPost ? (
          <>
            <p className="muted" style={{ margin: "2px 0 6px", fontSize: 12.5 }}>
              {entry.detail}
            </p>
            <PostLogLine stats={stats} />
            {loggedPost.planner_idea_id != null && (
              <button type="button" className="btn btn-sm" style={{ marginRight: 8 }} onClick={() => onOpenIdea(loggedPost.planner_idea_id!)}>
                Open post
              </button>
            )}
          </>
        ) : (
          entry.detail && (
            <p className="muted" style={{ margin: "2px 0 0", fontSize: 12.5, lineHeight: 1.55 }}>
              {entry.detail}
            </p>
          )
        )}
        {entry.noteTarget && <EntryNote key={`${entry.key}-${entry.note ?? ""}`} target={entry.noteTarget} note={entry.note ?? null} onChanged={onChanged} />}
        {deleteTarget && (
          <div style={{ marginTop: 8, display: "flex", gap: 10, alignItems: "center" }}>
            {confirmDelete ? (
              <>
                <span className="muted" style={{ fontSize: 12 }}>
                  Delete this {deleteTarget.what} for everyone?
                </span>
                <button type="button" className="link-btn danger" style={{ color: "var(--negative)" }} onClick={remove}>
                  Delete
                </button>
                <button type="button" className="link-btn" onClick={() => setConfirmDelete(false)}>
                  Keep
                </button>
              </>
            ) : (
              <button type="button" className="link-btn danger" onClick={() => setConfirmDelete(true)}>
                Delete {deleteTarget.what}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** How a posted item is doing, from the Post log. */
function PostLogLine({ stats }: { stats: string | null }) {
  return (
    <p style={{ margin: "0 0 8px", fontSize: 12.5 }}>
      <span style={{ color: "var(--second)" }}>Post log</span>
      <span className="muted"> · {stats ?? "no numbers yet; the next Meta import fills them in"} · </span>
      <a href="/posts" style={{ color: "var(--text-soft)" }}>
        Open Post log
      </a>
    </p>
  );
}

const NOTE_ENDPOINT: Record<NonNullable<CalendarEntry["noteTarget"]>["kind"], string> = {
  "trade-show": "/api/social-intelligence/trade-shows",
  "calendar-item": "/api/social-intelligence/calendar-items",
};

/** Add / edit / remove the free-form note on an event or a hand-added holiday/deadline. */
function EntryNote({ target, note, onChanged }: { target: NonNullable<CalendarEntry["noteTarget"]>; note: string | null; onChanged: () => void | Promise<void> }) {
  const [editing, setEditing] = useState(false);
  const [confirmingRemove, setConfirmingRemove] = useState(false);
  const [draft, setDraft] = useState(note ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save(notes: string | null) {
    setSaving(true);
    setError(null);
    const res = await fetch(`${NOTE_ENDPOINT[target.kind]}/${target.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notes }),
    }).catch(() => null);
    setSaving(false);
    if (!res?.ok) return setError(await readError(res, "Couldn't save the note."));
    setEditing(false);
    setConfirmingRemove(false);
    await onChanged();
  }

  if (editing) {
    return (
      <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 8 }}>
        <textarea className="input" rows={3} value={draft} onChange={(e) => setDraft(e.target.value)} aria-label="Note" autoFocus />
        <div style={{ display: "flex", gap: 8 }}>
          <button type="button" className="btn btn-sm btn-primary" disabled={saving} onClick={() => save(draft.trim() || null)}>
            {saving ? "Saving…" : "Save note"}
          </button>
          <button
            type="button"
            className="btn btn-sm btn-ghost"
            disabled={saving}
            onClick={() => {
              setDraft(note ?? "");
              setEditing(false);
            }}
          >
            Cancel
          </button>
        </div>
        {error && <p style={{ margin: 0, fontSize: 12, color: "var(--negative)" }}>{error}</p>}
      </div>
    );
  }

  if (!note) {
    return (
      <button type="button" className="link-btn" style={{ marginTop: 8, color: "var(--accent)", display: "block" }} onClick={() => setEditing(true)}>
        + Add note
      </button>
    );
  }

  return (
    <div className="card-quiet" style={{ marginTop: 10, padding: "10px 12px" }}>
      <p style={{ margin: 0, fontSize: 13, lineHeight: 1.55, whiteSpace: "pre-wrap" }}>{note}</p>
      <div style={{ display: "flex", gap: 12, marginTop: 6, alignItems: "center" }}>
        {confirmingRemove ? (
          <>
            <span className="muted" style={{ fontSize: 12 }}>
              Remove this note?
            </span>
            <button type="button" className="link-btn danger" style={{ color: "var(--negative)" }} disabled={saving} onClick={() => save(null)}>
              Remove
            </button>
            <button type="button" className="link-btn" onClick={() => setConfirmingRemove(false)}>
              Keep
            </button>
          </>
        ) : (
          <>
            <button
              type="button"
              className="link-btn"
              onClick={() => {
                setDraft(note);
                setEditing(true);
              }}
            >
              Edit note
            </button>
            <button type="button" className="link-btn danger" onClick={() => setConfirmingRemove(true)}>
              Remove note
            </button>
          </>
        )}
      </div>
      {error && <p style={{ margin: "6px 0 0", fontSize: 12, color: "var(--negative)" }}>{error}</p>}
    </div>
  );
}

type AddKind = "scheduled" | "deadline" | "posted" | "event" | "holiday";
const ADD_KINDS: { key: AddKind; label: string; hint: string }[] = [
  { key: "scheduled", label: "Scheduled post", hint: "A post planned for this day. It opens in the planner so the team can add the caption and comment." },
  { key: "deadline", label: "Post-by deadline", hint: "The last day to have something live. Shows in the Due soon alert two days before." },
  { key: "posted", label: "Posted", hint: "Something that already went out. It's added to the Post log too, and the next Meta import fills in its numbers." },
  { key: "event", label: "Event", hint: "A show, market or appearance. Its post-by deadline is worked out from the lead time." },
  { key: "holiday", label: "Holiday", hint: "A date worth planning around that isn't already on the calendar." },
];

function AddToCalendarForm({ initialDate, onCancel, onSaved }: { initialDate: string; onCancel: () => void; onSaved: () => void | Promise<void> }) {
  const [kind, setKind] = useState<AddKind>("scheduled");
  const [form, setForm] = useState({
    title: "",
    date: initialDate || (kind === "posted" ? todayIso() : ""),
    end_date: "",
    platform: "instagram",
    format: "photo",
    caption: "",
    location: "",
    lead_days: "10",
    notes: "",
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => setForm({ ...form, [k]: e.target.value });
  const isPost = kind === "scheduled" || kind === "posted";

  function pickKind(k: AddKind) {
    setKind(k);
    // Something that already went out usually went out today.
    if (k === "posted" && !form.date) setForm((f) => ({ ...f, date: todayIso() }));
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    const request =
      kind === "event"
        ? {
            url: "/api/social-intelligence/trade-shows",
            body: { name: form.title, location: form.location || null, start_date: form.date, end_date: form.end_date || null, lead_days: Number(form.lead_days) || 10, notes: form.notes || null },
          }
        : isPost
          ? {
              url: "/api/social-intelligence/content-ideas",
              body: { manual: true, product: form.title, target_date: form.date, platform: form.platform, format: form.format, caption: form.caption, status: kind === "posted" ? "used" : undefined },
            }
          : { url: "/api/social-intelligence/calendar-items", body: { kind: kind === "holiday" ? "holiday" : "deadline", title: form.title, date: form.date, notes: form.notes } };
    const res = await fetch(request.url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request.body) }).catch(() => null);
    setSaving(false);
    if (!res?.ok) return setError(await readError(res, "Couldn't add that to the calendar."));
    await onSaved();
  }

  const titleLabel = { scheduled: "Post title", posted: "Post title", deadline: "What needs to be live", event: "Event name", holiday: "Holiday name" }[kind];
  const titlePlaceholder = { scheduled: "e.g. Dragon kit before / after", posted: "e.g. Shell pendant carousel", deadline: "e.g. Holiday gift guide", event: "e.g. Circle Craft", holiday: "e.g. Studio anniversary" }[kind];

  return (
    <form onSubmit={submit} className="card-quiet" style={{ padding: 16, marginBottom: 16, display: "flex", flexDirection: "column", gap: 14 }}>
      <div role="radiogroup" aria-label="What are you adding?" style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
        {ADD_KINDS.map((k) => {
          const on = k.key === kind;
          return (
            <button
              key={k.key}
              type="button"
              role="radio"
              aria-checked={on}
              onClick={() => pickKind(k.key)}
              className="btn btn-sm"
              style={{ borderColor: on ? "var(--accent)" : "var(--border-strong)", background: on ? "var(--accent-soft)" : "transparent", color: on ? "var(--text)" : "var(--text-soft)" }}
            >
              <span aria-hidden style={{ width: 6, height: 6, borderRadius: "50%", background: DAY_KIND[k.key].color }} />
              {k.label}
            </button>
          );
        })}
      </div>
      <p className="muted" style={{ margin: 0, fontSize: 12.5 }}>
        {ADD_KINDS.find((k) => k.key === kind)!.hint}
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12 }}>
        <div className="field" style={{ gridColumn: "1 / -1" }}>
          <label htmlFor="add-title">{titleLabel}</label>
          <input id="add-title" className="input" required maxLength={120} placeholder={titlePlaceholder} value={form.title} onChange={set("title")} />
        </div>
        <div className="field">
          <label htmlFor="add-date">{kind === "event" ? "Starts" : kind === "posted" ? "Posted on" : "Date"}</label>
          <input id="add-date" className="input" type="date" required value={form.date} onChange={set("date")} />
        </div>
        {kind === "event" && (
          <>
            <div className="field">
              <label htmlFor="add-end">Ends</label>
              <input id="add-end" className="input" type="date" min={form.date || undefined} value={form.end_date} onChange={set("end_date")} />
            </div>
            <div className="field">
              <label htmlFor="add-location">Location</label>
              <input id="add-location" className="input" placeholder="Vancouver, BC" value={form.location} onChange={set("location")} />
            </div>
            <div className="field">
              <label htmlFor="add-lead">Post this many days before</label>
              <input id="add-lead" className="input" type="number" min={0} value={form.lead_days} onChange={set("lead_days")} />
            </div>
          </>
        )}
        {isPost && (
          <>
            <div className="field">
              <label htmlFor="add-platform">Platform</label>
              <select id="add-platform" className="input" value={form.platform} onChange={set("platform")}>
                {Object.entries(PLATFORM_LABEL).map(([v, l]) => (
                  <option key={v} value={v}>
                    {l}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="add-format">Format</label>
              <select id="add-format" className="input" value={form.format} onChange={set("format")}>
                {Object.entries(FORMAT_LABEL).map(([v, l]) => (
                  <option key={v} value={v}>
                    {l}
                  </option>
                ))}
              </select>
            </div>
            <div className="field" style={{ gridColumn: "1 / -1" }}>
              <label htmlFor="add-caption">Caption (optional)</label>
              <textarea id="add-caption" className="input" rows={3} value={form.caption} onChange={set("caption")} />
            </div>
          </>
        )}
        {!isPost && (
          <div className="field" style={{ gridColumn: "1 / -1" }}>
            <label htmlFor="add-notes">Note (optional)</label>
            <input id="add-notes" className="input" value={form.notes} onChange={set("notes")} />
          </div>
        )}
      </div>
      {error && <p style={{ margin: 0, fontSize: 12.5, color: "var(--negative)" }}>{error}</p>}
      <div style={{ display: "flex", gap: 8 }}>
        <button type="submit" className="btn btn-primary" disabled={saving || !form.title.trim() || !form.date}>
          {saving ? "Adding…" : `Add ${ADD_KINDS.find((k) => k.key === kind)!.label.toLowerCase()}`}
        </button>
        <button type="button" className="btn btn-ghost" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}

type IdeaFilter = "review" | "approved" | "unscheduled" | "posted";
const IDEA_FILTERS: { key: IdeaFilter; label: string; test: (i: PlannerIdea) => boolean }[] = [
  { key: "review", label: "To review", test: (i) => i.status === "suggested" },
  { key: "approved", label: "Approved", test: (i) => i.status === "approved" },
  { key: "unscheduled", label: "Unscheduled", test: (i) => !i.target_date && i.status !== "dismissed" && i.status !== "used" },
  { key: "posted", label: "Posted", test: (i) => i.status === "used" },
];

function IdeasCard({ ideas, loading, onOpen }: { ideas: PlannerIdea[]; loading: boolean; onOpen: (id: number) => void }) {
  const [filter, setFilter] = useState<IdeaFilter>("review");
  const active = IDEA_FILTERS.find((f) => f.key === filter)!;
  const list = ideas.filter(active.test).sort((a, b) => (a.target_date ?? "9999").localeCompare(b.target_date ?? "9999"));

  return (
    <section className="card" style={{ padding: "20px 22px" }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 12, flexWrap: "wrap", marginBottom: 6 }}>
        <div style={{ marginRight: "auto" }}>
          <h2 className="font-display" style={{ fontSize: 16, margin: "0 0 4px" }}>
            Content ideas
          </h2>
          <details className="about">
            <summary>About these ideas</summary>
            <p>
              Repost, refresh and new ideas written by asking Claude to act as a social media specialist against real post performance and
              real Shopify inventory. They aren&apos;t generated live by a button; ask for more anytime and they appear here for review.
            </p>
          </details>
        </div>
        <div role="tablist" aria-label="Filter ideas" style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {IDEA_FILTERS.map((f) => {
            const count = ideas.filter(f.test).length;
            const on = f.key === filter;
            return (
              <button
                key={f.key}
                role="tab"
                aria-selected={on}
                type="button"
                onClick={() => setFilter(f.key)}
                className="btn btn-sm"
                style={{ borderColor: on ? "var(--border-strong)" : "transparent", background: on ? "var(--surface-raised)" : "transparent", color: on ? "var(--text)" : "var(--text-soft)" }}
              >
                {f.label}
                <span className="tabular" style={{ color: "var(--text-faint)" }}>
                  {count}
                </span>
              </button>
            );
          })}
        </div>
      </div>
      {loading ? (
        <p className="muted">Loading…</p>
      ) : list.length === 0 ? (
        <p className="muted" style={{ fontSize: 13, margin: "12px 0 0" }}>
          Nothing here right now.
        </p>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(min(100%, 300px), 1fr))", gap: 12, marginTop: 12 }}>
          {list.map((idea) => (
            <button
              key={idea.id}
              type="button"
              onClick={() => onOpen(idea.id)}
              className="card-quiet"
              style={{ textAlign: "left", border: "1px solid transparent", padding: "14px 16px", cursor: "pointer", display: "flex", flexDirection: "column", gap: 8 }}
            >
              <span style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                {/* The filter tab already says the status, except on "Unscheduled", which mixes them. */}
                {filter === "unscheduled" && (
                  <span className={statusPillClass(idea.status)}>
                    <span className="dot" />
                    {STATUS_LABEL[idea.status]}
                  </span>
                )}
                <span style={{ fontSize: 12, color: "var(--text-faint)" }}>
                  {PLATFORM_LABEL[idea.platform] ?? idea.platform} · {FORMAT_LABEL[idea.format]}
                  {idea.target_date ? ` · ${shortDate(idea.target_date)}` : ""}
                </span>
              </span>
              <span style={{ fontSize: 14, fontWeight: 500 }}>{idea.product}</span>
              <span className="clamp-2" style={{ fontSize: 13, color: "var(--text-soft)", lineHeight: 1.5 }}>
                {idea.hook ? `“${idea.hook}” ` : ""}
                {idea.caption}
              </span>
              {idea.comments.length > 0 && (
                <span style={{ fontSize: 12, color: "var(--text-faint)" }}>
                  {idea.comments.length} comment{idea.comments.length === 1 ? "" : "s"} · last by {idea.comments[idea.comments.length - 1]!.author_name}
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
