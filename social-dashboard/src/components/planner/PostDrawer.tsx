"use client";

import { useEffect, useState } from "react";
import { POSTING_TIME_RESEARCH_NOTE } from "@/lib/social/postingTimes";
import {
  CONFIDENCE_LABEL,
  FORMAT_LABEL,
  PLATFORM_LABEL,
  STATUS_LABEL,
  postStats,
  readError,
  relativeTime,
  shortDate,
  statusPillClass,
  type ContentIdeaFormat,
  type ContentIdeaStatus,
  type Me,
  type PlannerComment,
  type PlannerIdea,
  type PlannerLoggedPost,
} from "./types";

export function Avatar({ name, size = 24 }: { name: string; size?: number }) {
  return (
    <span
      aria-hidden
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        background: "var(--surface-raised)",
        display: "inline-grid",
        placeItems: "center",
        fontSize: size * 0.45,
        fontWeight: 600,
        flexShrink: 0,
      }}
    >
      {name.charAt(0).toUpperCase()}
    </span>
  );
}

const STATUS_ACTIONS: { status: ContentIdeaStatus; label: string }[] = [
  { status: "approved", label: "Approve" },
  { status: "used", label: "Mark as posted" },
  { status: "dismissed", label: "Dismiss" },
  { status: "suggested", label: "Back to suggested" },
];

export function PostDrawer({
  idea,
  loggedPost,
  me,
  onClose,
  onChanged,
}: {
  idea: PlannerIdea;
  loggedPost?: PlannerLoggedPost;
  me: Me | null;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && !editing && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editing, onClose]);

  async function setStatus(status: ContentIdeaStatus) {
    setBusy(true);
    setStatusError(null);
    const res = await fetch(`/api/social-intelligence/content-ideas/${idea.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    }).catch(() => null);
    setBusy(false);
    if (!res?.ok) return setStatusError(await readError(res, "Couldn't change the status."));
    onChanged();
  }

  async function deletePost() {
    setBusy(true);
    const res = await fetch(`/api/social-intelligence/content-ideas/${idea.id}`, { method: "DELETE" }).catch(() => null);
    setBusy(false);
    if (!res?.ok) return setStatusError(await readError(res, "Couldn't delete this post."));
    onChanged();
    onClose();
  }

  const lastStatus = idea.activity.find((a) => a.action === "status");

  return (
    <div className="overlay" onClick={() => !editing && onClose()}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={idea.product} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
          <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              <span className={statusPillClass(idea.status)}>
                <span className="dot" />
                {STATUS_LABEL[idea.status]}
              </span>
              <span className="pill">{PLATFORM_LABEL[idea.platform] ?? idea.platform}</span>
              <span className="pill">{FORMAT_LABEL[idea.format]}</span>
            </div>
            <h2 className="font-display" style={{ fontSize: 20, margin: 0, lineHeight: 1.3 }}>
              {idea.product}
            </h2>
            <p className="muted" style={{ margin: 0, fontSize: 13 }} title={idea.suggested_time ? POSTING_TIME_RESEARCH_NOTE : undefined}>
              {idea.target_date ? `${shortDate(idea.target_date)}${idea.suggested_time ? ` · ${idea.suggested_time}` : ""}` : "Not scheduled yet"}
            </p>
          </div>
          <button type="button" className="btn btn-ghost btn-sm" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        {/* Status */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {STATUS_ACTIONS.filter((a) => a.status !== idea.status && !(a.status === "suggested" && idea.status === "suggested")).map((a) => (
              <button
                key={a.status}
                type="button"
                disabled={busy}
                className={`btn btn-sm ${a.status === "approved" && idea.status === "suggested" ? "btn-primary" : a.status === "dismissed" || a.status === "suggested" ? "btn-ghost" : ""}`}
                onClick={() => setStatus(a.status)}
              >
                {a.label}
              </button>
            ))}
          </div>
          {lastStatus && (
            <p className="muted" style={{ margin: 0, fontSize: 12 }}>
              {lastStatus.summary} by {lastStatus.author_name} · {relativeTime(lastStatus.created_at)}
            </p>
          )}
          {statusError && <p style={{ margin: 0, fontSize: 12, color: "var(--negative)" }}>{statusError}</p>}
          {idea.status === "used" && (
            <p className="card-quiet" style={{ margin: "4px 0 0", padding: "10px 12px", fontSize: 12.5 }}>
              <span style={{ color: "var(--second)" }}>In the Post log</span>
              <span className="muted">
                {" · "}
                {loggedPost ? (postStats(loggedPost) ?? "no numbers yet; the next Meta import fills them in") : "linking…"}
                {" · "}
              </span>
              <a href="/posts" style={{ color: "var(--text-soft)" }}>
                Open Post log
              </a>
            </p>
          )}
        </div>

        <hr className="divider" />

        {editing ? (
          <PostEditor
            idea={idea}
            onCancel={() => setEditing(false)}
            onSaved={() => {
              setEditing(false);
              onChanged();
            }}
          />
        ) : (
          <section style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", alignItems: "center" }}>
              <span className="eyebrow" style={{ marginRight: "auto" }}>
                Post
              </span>
              <button type="button" className="btn btn-sm" onClick={() => setEditing(true)}>
                Edit post
              </button>
            </div>
            {idea.hook && <p style={{ margin: 0, fontSize: 15, fontWeight: 500 }}>“{idea.hook}”</p>}
            {idea.caption ? (
              <p style={{ margin: 0, fontSize: 14, lineHeight: 1.65, whiteSpace: "pre-wrap" }}>{idea.caption}</p>
            ) : (
              <p className="muted" style={{ margin: 0, fontSize: 13.5 }}>
                No caption yet.
              </p>
            )}
            {idea.hashtags.length > 0 && <p style={{ margin: 0, fontSize: 13, color: "var(--text-soft)" }}>{idea.hashtags.map((h) => h.tag).join(" ")}</p>}
            {idea.cta && (
              <p style={{ margin: 0, fontSize: 13 }}>
                <span className="muted">Call to action · </span>
                {idea.cta}
              </p>
            )}
            <details className="about">
              <summary>Why this post</summary>
              <p>{idea.reasoning}</p>
              <p>
                {CONFIDENCE_LABEL[idea.confidence]}
                {idea.inventory_verified ? " · Inventory checked" : " · Inventory not checked"}
                {idea.pillar ? ` · Pillar: ${idea.pillar}` : ""}
              </p>
            </details>
          </section>
        )}

        <hr className="divider" />

        <CommentThread ideaId={idea.id} comments={idea.comments} me={me} onChanged={onChanged} />

        {idea.activity.length > 0 && (
          <>
            <hr className="divider" />
            <History entries={idea.activity} />
          </>
        )}

        <hr className="divider" />
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {confirmDelete ? (
            <>
              <span className="muted" style={{ fontSize: 13 }}>
                Delete this post for everyone? Its comments go too.
              </span>
              <button type="button" className="btn btn-sm btn-danger" disabled={busy} onClick={deletePost}>
                Delete
              </button>
              <button type="button" className="btn btn-sm btn-ghost" onClick={() => setConfirmDelete(false)}>
                Keep
              </button>
            </>
          ) : (
            <button type="button" className="link-btn danger" onClick={() => setConfirmDelete(true)}>
              Delete post
            </button>
          )}
        </div>
      </aside>
    </div>
  );
}

function PostEditor({ idea, onCancel, onSaved }: { idea: PlannerIdea; onCancel: () => void; onSaved: () => void }) {
  const [form, setForm] = useState({
    product: idea.product,
    target_date: idea.target_date ?? "",
    platform: idea.platform,
    format: idea.format as ContentIdeaFormat,
    pillar: idea.pillar ?? "",
    hook: idea.hook ?? "",
    caption: idea.caption,
    hashtags_text: idea.hashtags.map((h) => h.tag).join(" "),
    cta: idea.cta ?? "",
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => setForm({ ...form, [k]: e.target.value });

  async function save(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    const res = await fetch(`/api/social-intelligence/content-ideas/${idea.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...form, target_date: form.target_date || null }),
    }).catch(() => null);
    setSaving(false);
    if (!res?.ok) return setError(await readError(res, "Couldn't save your changes."));
    onSaved();
  }

  const id = (k: string) => `edit-${idea.id}-${k}`;
  return (
    <form onSubmit={save} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <span className="eyebrow">Edit post</span>
      <div className="field">
        <label htmlFor={id("product")}>Title</label>
        <input id={id("product")} className="input" required value={form.product} onChange={set("product")} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12 }}>
        <div className="field">
          <label htmlFor={id("date")}>Date</label>
          <input id={id("date")} type="date" className="input" value={form.target_date} onChange={set("target_date")} />
        </div>
        <div className="field">
          <label htmlFor={id("platform")}>Platform</label>
          <select id={id("platform")} className="input" value={form.platform} onChange={set("platform")}>
            {Object.entries(PLATFORM_LABEL).map(([v, l]) => (
              <option key={v} value={v}>
                {l}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor={id("format")}>Format</label>
          <select id={id("format")} className="input" value={form.format} onChange={set("format")}>
            {Object.entries(FORMAT_LABEL).map(([v, l]) => (
              <option key={v} value={v}>
                {l}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="field">
        <label htmlFor={id("hook")}>Hook</label>
        <input id={id("hook")} className="input" placeholder="The first line people see" value={form.hook} onChange={set("hook")} />
      </div>
      <div className="field">
        <label htmlFor={id("caption")}>Caption</label>
        <textarea id={id("caption")} className="input" rows={7} value={form.caption} onChange={set("caption")} />
      </div>
      <div className="field">
        <label htmlFor={id("hashtags")}>Hashtags</label>
        <input id={id("hashtags")} className="input" placeholder="#stonecarving #diycraftkit" value={form.hashtags_text} onChange={set("hashtags_text")} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
        <div className="field">
          <label htmlFor={id("cta")}>Call to action</label>
          <input id={id("cta")} className="input" value={form.cta} onChange={set("cta")} />
        </div>
        <div className="field">
          <label htmlFor={id("pillar")}>Pillar</label>
          <input id={id("pillar")} className="input" placeholder="e.g. behind-the-scenes" value={form.pillar} onChange={set("pillar")} />
        </div>
      </div>
      {error && <p style={{ margin: 0, fontSize: 12.5, color: "var(--negative)" }}>{error}</p>}
      <div style={{ display: "flex", gap: 8 }}>
        <button type="submit" className="btn btn-primary" disabled={saving}>
          {saving ? "Saving…" : "Save changes"}
        </button>
        <button type="button" className="btn btn-ghost" disabled={saving} onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}

function CommentThread({ ideaId, comments, me, onChanged }: { ideaId: number; comments: PlannerComment[]; me: Me | null; onChanged: () => void }) {
  const [draft, setDraft] = useState("");
  const [posting, setPosting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function post() {
    if (!draft.trim()) return;
    setPosting(true);
    setError(null);
    const res = await fetch(`/api/social-intelligence/content-ideas/${ideaId}/comments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ body: draft }),
    }).catch(() => null);
    setPosting(false);
    if (!res?.ok) return setError(await readError(res, "Couldn't post your comment."));
    setDraft("");
    onChanged();
  }

  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <span className="eyebrow">Comments{comments.length > 0 ? ` · ${comments.length}` : ""}</span>
      {comments.length === 0 && (
        <p className="muted" style={{ margin: 0, fontSize: 13 }}>
          No comments yet. Leave a note for the team: shoot ideas, a wording tweak, who&apos;s handling it.
        </p>
      )}
      {comments.map((c) => (
        <CommentRow key={`${c.id}-${c.edited_at ?? ""}`} comment={c} me={me} onChanged={onChanged} />
      ))}
      <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
        {me && <Avatar name={me.name} size={28} />}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
          <label htmlFor={`comment-${ideaId}`} style={{ position: "absolute", width: 1, height: 1, overflow: "hidden", clip: "rect(0 0 0 0)" }}>
            Add a comment
          </label>
          <textarea
            id={`comment-${ideaId}`}
            className="input"
            rows={2}
            placeholder={me ? `Comment as ${me.name}` : "Add a comment"}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) post();
            }}
          />
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button type="button" className="btn btn-sm btn-primary" disabled={posting || !draft.trim()} onClick={post}>
              {posting ? "Posting…" : "Comment"}
            </button>
            <span style={{ fontSize: 11.5, color: "var(--text-faint)" }}>⌘ Enter to send</span>
          </div>
          {error && <p style={{ margin: 0, fontSize: 12, color: "var(--negative)" }}>{error}</p>}
        </div>
      </div>
    </section>
  );
}

function CommentRow({ comment, me, onChanged }: { comment: PlannerComment; me: Me | null; onChanged: () => void }) {
  const [editing, setEditing] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [draft, setDraft] = useState(comment.body);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mine = me?.id === comment.person_id;
  const canDelete = mine;

  async function send(method: "PATCH" | "DELETE") {
    setBusy(true);
    setError(null);
    const res = await fetch(`/api/social-intelligence/comments/${comment.id}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: method === "PATCH" ? JSON.stringify({ body: draft }) : undefined,
    }).catch(() => null);
    setBusy(false);
    if (!res?.ok) return setError(await readError(res, method === "PATCH" ? "Couldn't save your edit." : "Couldn't delete that comment."));
    setEditing(false);
    setConfirmingDelete(false);
    onChanged();
  }

  return (
    <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
      <Avatar name={comment.author_name} size={28} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
          <strong style={{ fontSize: 13, fontWeight: 600 }}>{comment.author_name}</strong>
          <span style={{ fontSize: 12, color: "var(--text-faint)" }}>
            {relativeTime(comment.created_at)}
            {comment.edited_at ? " · edited" : ""}
          </span>
        </div>
        {editing ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 6 }}>
            <textarea className="input" rows={3} value={draft} onChange={(e) => setDraft(e.target.value)} aria-label="Edit comment" autoFocus />
            <div style={{ display: "flex", gap: 8 }}>
              <button type="button" className="btn btn-sm btn-primary" disabled={busy || !draft.trim()} onClick={() => send("PATCH")}>
                Save
              </button>
              <button
                type="button"
                className="btn btn-sm btn-ghost"
                onClick={() => {
                  setDraft(comment.body);
                  setEditing(false);
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <p style={{ margin: "3px 0 0", fontSize: 13.5, lineHeight: 1.55, whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{comment.body}</p>
        )}
        {!editing && (mine || canDelete) && (
          <div style={{ display: "flex", gap: 12, marginTop: 4, alignItems: "center" }}>
            {confirmingDelete ? (
              <>
                <span style={{ fontSize: 12, color: "var(--text-soft)" }}>Delete this comment?</span>
                <button type="button" className="link-btn danger" style={{ color: "var(--negative)" }} disabled={busy} onClick={() => send("DELETE")}>
                  Delete
                </button>
                <button type="button" className="link-btn" onClick={() => setConfirmingDelete(false)}>
                  Keep
                </button>
              </>
            ) : (
              <>
                {mine && (
                  <button type="button" className="link-btn" onClick={() => setEditing(true)}>
                    Edit
                  </button>
                )}
                {canDelete && (
                  <button type="button" className="link-btn danger" onClick={() => setConfirmingDelete(true)}>
                    Delete
                  </button>
                )}
              </>
            )}
          </div>
        )}
        {error && <p style={{ margin: "4px 0 0", fontSize: 12, color: "var(--negative)" }}>{error}</p>}
      </div>
    </div>
  );
}

const HISTORY_COLLAPSED = 4;

function History({ entries }: { entries: PlannerIdea["activity"] }) {
  const [all, setAll] = useState(false);
  const shown = all ? entries : entries.slice(0, HISTORY_COLLAPSED);
  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <span className="eyebrow">History</span>
      <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 8 }}>
        {shown.map((a) => (
          <li key={a.id} style={{ fontSize: 12.5, display: "flex", gap: 8 }}>
            <span style={{ color: "var(--text-faint)", flexShrink: 0, width: 64 }}>{relativeTime(a.created_at)}</span>
            <span>
              <strong style={{ fontWeight: 600 }}>{a.author_name}</strong> <span className="muted">· {a.summary}</span>
            </span>
          </li>
        ))}
      </ol>
      {entries.length > HISTORY_COLLAPSED && (
        <button type="button" className="link-btn" style={{ alignSelf: "flex-start" }} onClick={() => setAll((v) => !v)}>
          {all ? "Show less" : `Show all ${entries.length}`}
        </button>
      )}
    </section>
  );
}
