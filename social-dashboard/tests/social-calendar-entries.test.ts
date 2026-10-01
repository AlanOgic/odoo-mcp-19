import { describe, it, expect } from "vitest";
import { buildCalendarEntries, upcomingDeadlineAlerts, type CalendarOccasion, type CalendarTradeShow, type CalendarContentIdea } from "@/lib/social/calendarEntries";

const OCCASION: CalendarOccasion = {
  id: "halloween",
  name: "Halloween",
  date: "2026-10-31",
  suggestedPostByDate: "2026-10-17",
  note: "Test note",
};

const TRADE_SHOW: CalendarTradeShow = {
  id: 1,
  name: "Circle Craft",
  location: "Vancouver, BC",
  start_date: "2026-11-11",
  end_date: "2026-11-16",
  lead_days: 10,
  post_by_date: "2026-11-01",
  notes: null,
};

describe("buildCalendarEntries", () => {
  it("produces an entry for every day of a multi-day trade show", () => {
    const entries = buildCalendarEntries([], [TRADE_SHOW], []);
    const showDays = entries.filter((e) => e.type === "trade-show");
    expect(showDays).toHaveLength(6); // Nov 11-16 inclusive
    expect(showDays[0]!.date).toBe("2026-11-11");
    expect(showDays[5]!.date).toBe("2026-11-16");
  });

  it("produces a distinct post-deadline entry for a trade show", () => {
    const entries = buildCalendarEntries([], [TRADE_SHOW], []);
    const deadline = entries.find((e) => e.type === "post-deadline" && e.label.includes("Circle Craft"));
    expect(deadline).toBeDefined();
    expect(deadline!.date).toBe("2026-11-01");
  });

  it("produces occasion and post-deadline entries for a seasonal occasion", () => {
    const entries = buildCalendarEntries([OCCASION], [], []);
    expect(entries.find((e) => e.type === "occasion" && e.date === "2026-10-31")).toBeDefined();
    expect(entries.find((e) => e.type === "post-deadline" && e.date === "2026-10-17")).toBeDefined();
  });

  it("sorts all entries chronologically", () => {
    const entries = buildCalendarEntries([OCCASION], [TRADE_SHOW], []);
    const dates = entries.map((e) => e.date);
    expect(dates).toEqual([...dates].sort());
  });

  it("categorizes a story-format content idea as story-post, not content-idea", () => {
    const storyIdea: CalendarContentIdea = { id: 1, idea_type: "new", target_date: "2026-10-30", product: "Test Show", platform: "instagram", status: "suggested", format: "story" };
    const photoIdea: CalendarContentIdea = { id: 2, idea_type: "new", target_date: "2026-10-30", product: "Test Photo", platform: "instagram", status: "suggested", format: "photo" };
    const entries = buildCalendarEntries([], [], [storyIdea, photoIdea]);
    expect(entries.find((e) => e.key === "content-idea-1")!.type).toBe("story-post");
    expect(entries.find((e) => e.key === "content-idea-2")!.type).toBe("content-idea");
  });

  it("points a scheduled post at its own content idea so the calendar can open it", () => {
    const idea: CalendarContentIdea = { id: 7, idea_type: "new", target_date: "2026-10-30", product: "Test", platform: "instagram", status: "suggested", format: "photo" };
    const entry = buildCalendarEntries([], [], [idea]).find((e) => e.key === "content-idea-7")!;
    expect(entry.ideaId).toBe(7);
    expect(entry.noteTarget).toBeUndefined();
  });

  it("carries a trade show's notes as its editable note on every day, not inside the detail text", () => {
    const show: CalendarTradeShow = { ...TRADE_SHOW, notes: "Booth 214" };
    const showDays = buildCalendarEntries([], [show], []).filter((e) => e.type === "trade-show");
    for (const day of showDays) {
      expect(day.noteTarget).toEqual({ kind: "trade-show", id: 1 });
      expect(day.note).toBe("Booth 214");
      expect(day.detail).not.toContain("Booth 214");
    }
  });

  it("gives occasions and post-by deadlines no editable note", () => {
    const entries = buildCalendarEntries([OCCASION], [TRADE_SHOW], []);
    for (const e of entries.filter((x) => x.type === "occasion" || x.type === "post-deadline")) {
      expect(e.noteTarget).toBeUndefined();
    }
  });
});

describe("upcomingDeadlineAlerts", () => {
  it("includes a deadline exactly 2 days out", () => {
    const entries = buildCalendarEntries([OCCASION], [], []);
    const alerts = upcomingDeadlineAlerts(entries, 2, new Date("2026-10-15T00:00:00"));
    expect(alerts.some((a) => a.date === "2026-10-17")).toBe(true);
  });

  it("excludes a deadline 3 days out when the window is 2 days", () => {
    const entries = buildCalendarEntries([OCCASION], [], []);
    const alerts = upcomingDeadlineAlerts(entries, 2, new Date("2026-10-14T00:00:00"));
    expect(alerts.some((a) => a.date === "2026-10-17")).toBe(false);
  });

  it("excludes a deadline that has already passed", () => {
    const entries = buildCalendarEntries([OCCASION], [], []);
    const alerts = upcomingDeadlineAlerts(entries, 2, new Date("2026-10-20T00:00:00"));
    expect(alerts.some((a) => a.date === "2026-10-17")).toBe(false);
  });

  it("includes a deadline that is today", () => {
    const entries = buildCalendarEntries([OCCASION], [], []);
    const alerts = upcomingDeadlineAlerts(entries, 2, new Date("2026-10-17T00:00:00"));
    expect(alerts.some((a) => a.date === "2026-10-17")).toBe(true);
  });

  it("never includes a non-deadline entry", () => {
    const entries = buildCalendarEntries([OCCASION], [TRADE_SHOW], []);
    const alerts = upcomingDeadlineAlerts(entries, 365, new Date("2026-01-01T00:00:00"));
    expect(alerts.every((a) => a.type === "post-deadline")).toBe(true);
  });
});

describe("hand-added items and the Post log", () => {
  const LOGGED = { id: 8, posted_date: "2026-09-24", platform: "instagram", post_type: "IG carousel", caption: "Take yourself back to the shore. 🤍\nThe sound of the water." };

  it("shows Post log posts on their day, titled by the caption's first line", () => {
    const entry = buildCalendarEntries([], [], [], [], [LOGGED]).find((e) => e.loggedPostId === 8)!;
    expect(entry.type).toBe("logged-post");
    expect(entry.date).toBe("2026-09-24");
    expect(entry.label).toBe("Take yourself back to the shore. 🤍");
  });

  it("hides a Post log post that a posted planner item already stands for", () => {
    const linked: CalendarContentIdea = { id: 2, idea_type: "new", target_date: "2026-09-24", product: "Shell", platform: "instagram", status: "used", format: "carousel", posted_post_id: 8 };
    expect(buildCalendarEntries([], [], [linked], [], [LOGGED]).some((e) => e.loggedPostId === 8)).toBe(false);
    const unlinked: CalendarContentIdea = { ...linked, posted_post_id: null };
    expect(buildCalendarEntries([], [], [unlinked], [], [LOGGED]).some((e) => e.loggedPostId === 8)).toBe(false);
    const otherPlatform: CalendarContentIdea = { ...unlinked, platform: "facebook" };
    expect(buildCalendarEntries([], [], [otherPlatform], [], [LOGGED]).some((e) => e.loggedPostId === 8)).toBe(true);
  });

  it("turns hand-added holidays and deadlines into movable, noteable entries that feed deadline alerts", () => {
    const entries = buildCalendarEntries(
      [],
      [],
      [],
      [
        { id: 1, kind: "holiday", title: "Studio anniversary", date: "2026-10-10", notes: null },
        { id: 2, kind: "deadline", title: "Holiday gift guide", date: "2026-10-12", notes: "Needs the new kits" },
      ],
    );
    const holiday = entries.find((e) => e.customItemId === 1)!;
    expect(holiday.type).toBe("occasion");
    const deadline = entries.find((e) => e.customItemId === 2)!;
    expect(deadline.type).toBe("post-deadline");
    expect(deadline.noteTarget).toEqual({ kind: "calendar-item", id: 2 });
    expect(deadline.note).toBe("Needs the new kits");
    expect(upcomingDeadlineAlerts(entries, 2, new Date("2026-10-10T12:00:00")).map((e) => e.customItemId)).toEqual([2]);
  });
});
