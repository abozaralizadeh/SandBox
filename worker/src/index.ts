// SandBox's data URLs, answered at the edge.
//
// The pages are static files on GitHub Pages, but they still call the URLs the old Flask app
// served (/tomorrownewscontent, /aiblogindex, /traide/live, ...). Cloudflare routes exactly
// those paths to this Worker; everything else goes straight to Pages.
//
// Two kinds of data:
//  * Daily content (TomorrowNews, AIBlog, ComicBook, GenBox, AIOPS). A GitHub Actions job makes
//    it and `sitepub.py` copies what the pages read into the public `sandbox-site` container.
//    This Worker only maps a request to one of those files. It never generates anything.
//  * trAIde. The bot writes its own table and `live.json`, so this Worker reads them live with a
//    read-only SAS that covers only those two (TRAIDE_TABLE_SAS, TRAIDE_BLOB_SAS).
//
// It also has one cron (wrangler.toml) that starts the Generate workflow on GitHub every hour,
// because GitHub's own scheduler runs this account's crons hours late or skips them.
//
// Answers are cached at the edge (Cache API), so a busy page costs one storage read per TTL
// per Cloudflare location, not one per visitor.

export interface Env {
  BLOB_BASE: string; // https://<account>.blob.core.windows.net/sandbox-site
  COMIC_CONTAINER_URL: string; // https://<account>.blob.core.windows.net/comicbook-html
  TRAIDE_TABLE_URL: string; // https://<account>.table.core.windows.net/traidedashboard
  TRAIDE_BLOB_URL: string; // https://<account>.blob.core.windows.net/traide-dashboard
  AIOPS_DEFAULT_PROBLEM: string;
  TRAIDE_TABLE_SAS?: string; // secret, read-only on the trAIde table
  TRAIDE_BLOB_SAS?: string; // secret, read-only on the trAIde container
  GH_REPO: string; // owner/name of the repo whose Generate workflow the cron starts
  GH_DISPATCH_TOKEN?: string; // secret, fine-grained token: Actions read/write on GH_REPO only
}

type Handler = (req: Request, url: URL, env: Env) => Promise<Response>;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const pad = (n: number, w = 2) => String(n).padStart(w, "0");

function json(body: unknown, status = 200, maxAge = 60): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json",
      "cache-control": `public, max-age=${maxAge}`,
    },
  });
}

function text(body: string, status = 200, headers: Record<string, string> = {}, maxAge = 60): Response {
  return new Response(body, {
    status,
    headers: {
      "content-type": "text/html; charset=utf-8",
      "cache-control": `public, max-age=${maxAge}`,
      ...headers,
    },
  });
}

const notFound = () => text("404 Not Found", 404);

/** Read one published file. null when it does not exist. */
async function blob(env: Env, path: string, ttl = 60): Promise<string | null> {
  const res = await fetch(`${env.BLOB_BASE}/${path}`, { cf: { cacheTtl: ttl, cacheEverything: true } });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`blob ${path}: HTTP ${res.status}`);
  return res.text();
}

async function blobJson<T>(env: Env, path: string, ttl = 60): Promise<T | null> {
  const body = await blob(env, path, ttl);
  return body === null ? null : (JSON.parse(body) as T);
}

// ---------------------------------------------------------------------------
// Dates. The pages send the reader's LOCAL time ("2026-11-23T14:32:10.123"), and Flask parsed
// it with datetime.fromisoformat, keeping the wall-clock fields as given. Same here: the fields
// are read as written, never shifted to UTC.
// ---------------------------------------------------------------------------

export interface Wall {
  y: number;
  m: number;
  d: number;
  h: number;
  mi: number;
  s: number;
}

const ISO_RE = /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2})(?::(\d{2})(?::(\d{2})(?:\.\d+)?)?)?)?(?:Z|[+-]\d{2}:?\d{2})?$/;

/** Python's datetime.fromisoformat, as far as these pages use it. null when it would raise. */
export function parseWall(value: string | null): Wall | null {
  if (!value) return null;
  const m = ISO_RE.exec(value.trim());
  if (!m) return null;
  const w = { y: +m[1], m: +m[2], d: +m[3], h: +(m[4] ?? 0), mi: +(m[5] ?? 0), s: +(m[6] ?? 0) };
  if (w.m < 1 || w.m > 12 || w.d < 1 || w.d > 31 || w.h > 23 || w.mi > 59 || w.s > 59) return null;
  return w;
}

export function nowUtc(): Wall {
  const n = new Date();
  return {
    y: n.getUTCFullYear(), m: n.getUTCMonth() + 1, d: n.getUTCDate(),
    h: n.getUTCHours(), mi: n.getUTCMinutes(), s: n.getUTCSeconds(),
  };
}

export const flatDate = (w: Wall) => `${w.y}${pad(w.m)}${pad(w.d)}`;

/** "YYYYMMDD" or "YYYYMMDD_HH" -> the "YYYY-MM-DD HH:00:00" Flask sent as the Timestamp header. */
export function stamp(key: string): string {
  const hh = key.length > 8 ? key.slice(9, 11) : "00";
  return `${key.slice(0, 4)}-${key.slice(4, 6)}-${key.slice(6, 8)} ${hh}:00:00`;
}

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
  "September", "October", "November", "December"];

/** main.py's _missing_edition_page, for a day with no edition. */
export function missingEditionPage(title: string, key: string): string {
  const label = `${key.slice(6, 8)} ${MONTHS[+key.slice(4, 6) - 1]} ${key.slice(0, 4)}`;
  return `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${title}</title></head>
<body style="margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
             background:transparent;color:#444;display:flex;align-items:center;
             justify-content:center;min-height:60vh;text-align:center">
  <div style="padding:32px">
    <div style="font-size:34px;margin-bottom:12px">📭</div>
    <h2 style="margin:0 0 8px;font-size:20px;font-weight:600">No edition for ${label}</h2>
    <p style="margin:0;font-size:15px;line-height:1.5;color:#666">
      ${title} publishes once a day and was offline on this date, so nothing was written.<br>
      Use the arrows to jump to the closest date that has one.</p>
  </div>
</body></html>`;
}

/** "2026-11-23T00:00:00" (an index entry) -> "20261123_00". */
const isoToKey = (iso: string) => `${iso.slice(0, 4)}${iso.slice(5, 7)}${iso.slice(8, 10)}_${iso.slice(11, 13) || "00"}`;

// ---------------------------------------------------------------------------
// TomorrowNews
// ---------------------------------------------------------------------------

const TN_LANGS = ["en", "fa", "it"];

/** TomorrowNews.prompt._get_rowkey_base: daily from 2025-01-25, hourly before. */
export function tnBase(w: Wall | null): string {
  const t = w ?? nowUtc();
  if (w && flatDate(w) < "20250125") return `${flatDate(w)}_${pad(w.h)}`;
  return `${flatDate(t)}_00`;
}

interface TnIndex {
  editions: Record<string, string[]>;
  latest: Record<string, string>;
}

const tomorrownewsContent: Handler = async (req, url, env) => {
  if (!req.headers.get("referer")) return notFound();
  let lang = url.searchParams.get("lang") || "en";
  if (!TN_LANGS.includes(lang)) lang = "en";
  const dt = parseWall(url.searchParams.get("dt"));
  const base = tnBase(dt);

  const html = await blob(env, `tomorrownews/${base}_${lang}.html`, 300);
  if (html !== null) return text(html, 200, { Timestamp: stamp(base), "Edition-Missing": "0" }, 300);

  // Not published. A date past the newest edition is "today, before its run" (or the reader's
  // clock is ahead of UTC): show the newest edition under its own date, which the page then
  // displays. An older date is a real gap.
  const index = await blobJson<TnIndex>(env, "tomorrownews/index.json");
  const latest = index?.latest?.[lang];
  if (latest && (!dt || base > isoToKey(latest))) {
    const lbase = isoToKey(latest);
    const lhtml = await blob(env, `tomorrownews/${lbase}_${lang}.html`, 300);
    if (lhtml !== null) return text(lhtml, 200, { Timestamp: stamp(lbase), "Edition-Missing": "0" }, 60);
  }
  return text(missingEditionPage("Tomorrow News", base), 200,
    { Timestamp: stamp(base), "Edition-Missing": "1" }, 60);
};

const tomorrownewsIndex: Handler = async (req, _url, env) => {
  if (!req.headers.get("referer")) return notFound();
  const index = await blob(env, "tomorrownews/index.json");
  return index === null ? json({ editions: {}, latest: {} }) : new Response(index, { headers: jsonHeaders(60) });
};

const jsonHeaders = (maxAge: number) => ({
  "content-type": "application/json",
  "cache-control": `public, max-age=${maxAge}`,
});

// ---------------------------------------------------------------------------
// AIBlog
// ---------------------------------------------------------------------------

const aiblogContent: Handler = async (req, url, env) => {
  if (!req.headers.get("referer")) return notFound();
  const dt = parseWall(url.searchParams.get("dt"));
  const key = flatDate(dt ?? nowUtc());

  const html = await blob(env, `aiblog/${key}.html`, 300);
  if (html !== null) return text(html, 200, { Timestamp: stamp(key), "Edition-Missing": "0" }, 300);

  const index = await blobJson<{ latest?: string }>(env, "aiblog/index.json");
  const latest = index?.latest ? isoToKey(index.latest).slice(0, 8) : null;
  if (latest && (!dt || key > latest)) {
    const lhtml = await blob(env, `aiblog/${latest}.html`, 300);
    if (lhtml !== null) return text(lhtml, 200, { Timestamp: stamp(latest), "Edition-Missing": "0" }, 60);
  }
  return text(missingEditionPage("AI Blog", key), 200, { Timestamp: stamp(key), "Edition-Missing": "1" }, 60);
};

const aiblogIndex: Handler = async (req, _url, env) => {
  if (!req.headers.get("referer")) return notFound();
  const index = await blob(env, "aiblog/index.json");
  return index === null ? json({ editions: [], latest: null }) : new Response(index, { headers: jsonHeaders(60) });
};

// ---------------------------------------------------------------------------
// ComicBook
// ---------------------------------------------------------------------------

const CB_LANGS = ["en", "it", "fa"];
const NO_EPISODE_HTML =
  "<p style='color:#9aa;padding:24px 0;text-align:center'>No episode was published on this date.</p>";
const NOT_YET_HTML =
  "<p style='color:#9aa;padding:24px 0;text-align:center'>This episode is not out yet. Please check back later today.</p>";

/** A published page starts with "<!--arc:ID-->\n"; split it into the arc id and the HTML. */
export function splitArc(file: string): [string, string] {
  const m = /^<!--arc:([^>]*)-->\n/.exec(file);
  return m ? [m[1], file.slice(m[0].length)] : ["", file];
}

const comicbookContent: Handler = async (req, url, env) => {
  if (!req.headers.get("referer")) return notFound();
  let lang = url.searchParams.get("lang") || "en";
  if (!CB_LANGS.includes(lang)) lang = "en";
  const dt = parseWall(url.searchParams.get("dt"));
  const key = flatDate(dt ?? nowUtc());
  const dtStamp = dt ? `${dt.y}-${pad(dt.m)}-${pad(dt.d)} ${pad(dt.h)}:${pad(dt.mi)}:${pad(dt.s)}` : "";

  const file = await blob(env, `comicbook/${key}_${lang}.html`, 300);
  if (file !== null) {
    const [arc, html] = splitArc(file);
    // Without a date the page reads the day back from this header in UTC.
    const ts = dtStamp || `${key.slice(0, 4)}-${key.slice(4, 6)}-${key.slice(6, 8)}T00:00:00Z`;
    return text(html, 200, { Timestamp: ts, "Arc-Id": arc }, 300);
  }

  const index = await blobJson<{ episodes: { date: string }[] }>(env, "comicbook/index.json");
  const dates = (index?.episodes ?? []).map((e) => e.date).sort();
  const latest = dates[dates.length - 1];
  if (!dt && latest) {
    // First visit, today's episode not drawn yet: open on the newest one. The page takes the
    // date from Timestamp, so the label matches what is shown.
    const lfile = await blob(env, `comicbook/${latest}_${lang}.html`, 300);
    if (lfile !== null) {
      const [arc, html] = splitArc(lfile);
      const ts = `${latest.slice(0, 4)}-${latest.slice(4, 6)}-${latest.slice(6, 8)}T00:00:00Z`;
      return text(html, 200, { Timestamp: ts, "Arc-Id": arc }, 60);
    }
  }
  // With a date picked the page keeps that date on screen, so never swap in another day.
  const body = latest && key > latest ? NOT_YET_HTML : NO_EPISODE_HTML;
  return text(body, 200, { Timestamp: dtStamp || stamp(key), "Arc-Id": "" }, 60);
};

const comicbookIndex: Handler = async (_req, _url, env) => {
  const index = await blob(env, "comicbook/index.json");
  return index === null ? json({ episodes: [], arcs: [] }) : new Response(index, { headers: jsonHeaders(60) });
};

/** ComicBook.imageproxy._derived_name. */
export function derivedName(blobName: string, width: number | null, quality: number): string {
  const stem = blobName.replace(/\.[^.]*$/, "");
  return `${stem}${width ? `_w${width}` : ""}_q${quality}.webp`;
}

const cbimg: Handler = async (_req, url, env) => {
  // The publisher transcodes every panel when it publishes an episode, so this is only a
  // redirect. It still refuses anything outside the comic container (no open relay).
  const u = url.searchParams.get("u") || "";
  const prefix = `${env.COMIC_CONTAINER_URL}/`;
  if (!u.startsWith(prefix)) return text("Not found", 404);
  const name = u.slice(prefix.length).split("?")[0];
  let q = parseInt(url.searchParams.get("q") || "92", 10);
  q = Number.isFinite(q) ? Math.max(1, Math.min(100, q)) : 92;
  let w: number | null = parseInt(url.searchParams.get("w") || "", 10);
  w = Number.isFinite(w) ? Math.max(1, Math.min(4096, w as number)) : null;
  const webp = prefix + derivedName(name, w, q);
  const head = await fetch(webp, { method: "HEAD", cf: { cacheTtl: 86400, cacheEverything: true } });
  if (head.ok) {
    return new Response(null, {
      status: 302,
      headers: { location: webp, "cache-control": "public, max-age=31536000, immutable" },
    });
  }
  // No WebP (yet): the original PNG, but only cached briefly so the WebP wins once it exists.
  return new Response(null, { status: 302, headers: { location: prefix + name, "cache-control": "public, max-age=3600" } });
};

// ---------------------------------------------------------------------------
// GenBox
// ---------------------------------------------------------------------------

interface Channels {
  dates: string[];
  live: string;
  interval_days: number;
}

/** GenBox's ?date=YYYY-MM-DD -> "YYYYMMDD". Flask parsed it with date.fromisoformat. */
function gbDate(value: string | null): string | null | "invalid" {
  if (!value) return null;
  const w = parseWall(value);
  return w ? flatDate(w) : "invalid";
}

const getString: Handler = async (_req, url, env) => {
  let flat = gbDate(url.searchParams.get("date"));
  if (flat === "invalid") {
    return json({ error: "Invalid date format. Use ISO 8601 format, e.g., YYYY-MM-DDTHH:MM:SS" }, 400);
  }
  if (!flat) {
    const ch = await blobJson<Channels>(env, "genbox/channels.json");
    if (!ch?.live) return text("");
    flat = ch.live.replace(/-/g, "");
  }
  // No bulletin for that day -> "" and the TV shows static, as before.
  return text((await blob(env, `genbox/text/${flat}.txt`, 300)) ?? "", 200, {}, 120);
};

const genboxChannels: Handler = async (req, _url, env) => {
  if (!req.headers.get("referer")) return notFound();
  const ch = await blob(env, "genbox/channels.json");
  return ch === null ? json({ dates: [], live: "", interval_days: 1 }) : new Response(ch, { headers: jsonHeaders(60) });
};

const ABSENT = { audio_status: "absent", audio_url: "", status: "absent", video_url: "" };

async function genboxStatus(env: Env, value: string | null) {
  const flat = gbDate(value);
  const day = flat && flat !== "invalid" ? flat : flatDate(nowUtc());
  return (await blobJson<typeof ABSENT>(env, `genbox/status/${day}.json`)) ?? ABSENT;
}

const genboxVideoStatus: Handler = async (_req, url, env) => json(await genboxStatus(env, url.searchParams.get("date")));

/** The media is in a public container, so a redirect is enough (Range requests go there). */
const genboxMedia = (kind: "video" | "audio"): Handler => async (_req, url, env) => {
  const st = await genboxStatus(env, url.searchParams.get("date"));
  const ready = kind === "video" ? st.status === "ready" && st.video_url : st.audio_status === "ready" && st.audio_url;
  if (!ready) return text(kind === "video" ? "Video not ready" : "Audio not ready", 404);
  const target = kind === "video" ? st.video_url : st.audio_url;
  return new Response(null, { status: 302, headers: { location: target, "cache-control": "no-store" } });
};

// ---------------------------------------------------------------------------
// AI Open Problem Solver
// ---------------------------------------------------------------------------

/** AIOpenProblemSolver.azurestorage._problem_partition: sha1 of the lower-cased name. */
export async function problemSlug(problem: string): Promise<string> {
  const normalized = problem.trim().toLowerCase().split(/\s+/).filter(Boolean).join(" ");
  const digest = await crypto.subtle.digest("SHA-1", new TextEncoder().encode(normalized));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

const aiopsProblems: Handler = async (_req, _url, env) => {
  const body = await blob(env, "aiops/problems.json", 300);
  return body === null ? json({ problems: [] }) : new Response(body, { headers: jsonHeaders(300) });
};

const aiopsDetails: Handler = async (_req, url, env) => {
  const problem = url.searchParams.get("problem");
  if (!problem) return json({ error: "Missing 'problem' parameter." }, 400);
  const body = await blob(env, `aiops/${await problemSlug(problem)}/details.json`);
  if (body === null) return json({ problem, description: "", progress_percent: null, progress_comment: "" });
  return new Response(body, { headers: jsonHeaders(60) });
};

const INT_RE = /^\s*[+-]?\d+\s*$/;

const aiopsHistory: Handler = async (_req, url, env) => {
  const problem = url.searchParams.get("problem") || env.AIOPS_DEFAULT_PROBLEM;
  if (!problem) return json({ error: "Missing 'problem' parameter." }, 400);
  const rawOffset = url.searchParams.get("offset") ?? "0";
  const rawLimit = url.searchParams.get("limit") ?? "5";
  if (!INT_RE.test(rawOffset) || !INT_RE.test(rawLimit)) {
    return json({ error: "Offset and limit must be integers." }, 400);
  }
  const offset = Math.max(0, parseInt(rawOffset, 10));
  const limit = Math.max(1, Math.min(parseInt(rawLimit, 10), 10));

  const slug = await problemSlug(problem);
  const keys = (await blobJson<string[]>(env, `aiops/${slug}/keys.json`)) ?? [];
  const end = offset + limit;
  const window = keys.slice(offset, end);
  const entries = (
    await Promise.all(window.map((k) => blobJson<Record<string, unknown>>(env, `aiops/${slug}/e/${k}.json`, 300)))
  ).filter((e): e is Record<string, unknown> => e !== null);

  const first = offset === 0 ? entries[0] : undefined;
  return json({
    problem,
    entries,
    next_offset: end < keys.length ? end : null,
    progress_percent: first ? first.progress_percent ?? null : null,
    progress_comment: first ? first.progress_comment ?? "" : "",
    // Generation is scheduled now, never started by a visit, so nothing is ever "running
    // because you opened the page".
    generating: false,
  });
};

// ---------------------------------------------------------------------------
// trAIde (live, read with the read-only SAS)
// ---------------------------------------------------------------------------

const traideGuard = (req: Request) => (req.headers.get("referer") || "").includes("/traide");
const forbidden = () => json({ error: "forbidden" }, 403);

type Entity = Record<string, unknown> & { RowKey?: string };

/** Query the trAIde table, following continuation tokens. */
async function tableQuery(env: Env, filter: string, select?: string[]): Promise<Entity[]> {
  if (!env.TRAIDE_TABLE_SAS) return [];
  const out: Entity[] = [];
  let nextPk: string | null = null;
  let nextRk: string | null = null;
  for (let page = 0; page < 200; page++) {
    const qs = new URLSearchParams(env.TRAIDE_TABLE_SAS.replace(/^\?/, ""));
    qs.set("$filter", filter);
    if (select) qs.set("$select", select.join(","));
    if (nextPk) qs.set("NextPartitionKey", nextPk);
    if (nextRk) qs.set("NextRowKey", nextRk);
    const res = await fetch(`${env.TRAIDE_TABLE_URL}()?${qs}`, {
      headers: { accept: "application/json;odata=nometadata", "x-ms-version": "2020-12-06" },
    });
    if (!res.ok) throw new Error(`table: HTTP ${res.status}`);
    const body = (await res.json()) as { value: Entity[] };
    out.push(...body.value);
    nextPk = res.headers.get("x-ms-continuation-NextPartitionKey");
    nextRk = res.headers.get("x-ms-continuation-NextRowKey");
    if (!nextPk) break;
  }
  return out;
}

/** Rows whose `data` property holds one JSON object; newest (`ts`) first. */
function dataItems(rows: Entity[]): Record<string, unknown>[] {
  const items: Record<string, unknown>[] = [];
  for (const r of rows) {
    if (typeof r.data !== "string" || !r.data) continue;
    try {
      items.push(JSON.parse(r.data));
    } catch {
      /* a malformed row is skipped, as in Python */
    }
  }
  items.sort((a, b) => Number(b.ts ?? 0) - Number(a.ts ?? 0));
  return items;
}

const epochFloor = (days: number) => String(Math.floor(Date.now() / 1000) - Math.max(1, days) * 86400);
const FEED_WINDOW_DAYS = [3, 21, 90];

async function windowedPartition(env: Env, pk: string, limit: number, floors: (string | null)[]): Promise<unknown[]> {
  for (const floor of floors) {
    const filter = `PartitionKey eq '${pk}'` + (floor ? ` and RowKey ge '${floor}'` : "");
    const items = dataItems(await tableQuery(env, filter, ["data"]));
    if (!floor || items.length >= limit) return items.slice(0, limit);
  }
  return [];
}

function clampInt(raw: string | null, dflt: number, lo: number, hi: number): number {
  const n = raw !== null && INT_RE.test(raw) ? parseInt(raw, 10) : dflt;
  return Math.max(lo, Math.min(n, hi));
}

const traideLive: Handler = async (req, _url, env) => {
  if (!traideGuard(req)) return forbidden();
  if (!env.TRAIDE_BLOB_SAS) return json({}, 200, 30);
  const sas = env.TRAIDE_BLOB_SAS.replace(/^\?/, "");
  const res = await fetch(`${env.TRAIDE_BLOB_URL}/live.json?${sas}`);
  if (!res.ok) return json({}, 200, 30);
  return new Response(res.body, { headers: jsonHeaders(30) });
};

const INDEX_BASE = 100;
const INDEX_SANITY_FACTOR = 1000;

const traideEquity: Handler = async (req, url, env) => {
  if (!traideGuard(req)) return forbidden();
  const period = (url.searchParams.get("period") || "all").toLowerCase();
  const span = ({ day: 1, week: 7, month: 31, all: null } as Record<string, number | null>)[period] ?? null;
  const today = Math.floor(Date.now() / 86400000);
  let filter = "PartitionKey eq 'equity'";
  if (span) filter += ` and RowKey ge '${String(today - span).padStart(8, "0")}'`;
  const lo = INDEX_BASE / INDEX_SANITY_FACTOR;
  const hi = INDEX_BASE * INDEX_SANITY_FACTOR;
  const points: Record<string, unknown>[] = [];
  for (const r of await tableQuery(env, filter)) {
    const day = parseInt(String(r.RowKey ?? ""), 10);
    if (!Number.isFinite(day)) continue;
    const close = r.indexClose;
    if (close !== undefined && close !== null) {
      const f = Number(close);
      if (!Number.isFinite(f) || f < lo || f > hi) continue; // corrupt point, hidden as in Python
    }
    const point: Record<string, unknown> = { day, indexClose: close ?? null, drawdownPct: r.drawdownPct ?? null };
    if ("dayRealizedPnl" in r) point.dayRealizedPnl = r.dayRealizedPnl;
    points.push(point);
  }
  points.sort((a, b) => (a.day as number) - (b.day as number));
  return json({ period, points }, 200, 45);
};

const traideFeed: Handler = async (req, url, env) => {
  if (!traideGuard(req)) return forbidden();
  const limit = clampInt(url.searchParams.get("limit"), 30, 1, 100);
  const items = await windowedPartition(env, "decision", limit, [...FEED_WINDOW_DAYS.map(epochFloor), null]);
  return json({ items }, 200, 300);
};

const traideTrades: Handler = async (req, url, env) => {
  if (!traideGuard(req)) return forbidden();
  const limit = clampInt(url.searchParams.get("limit"), 100, 1, 200);
  const items = dataItems(await tableQuery(env, "PartitionKey eq 'trade'", ["data"]));
  return json({ items: items.slice(0, limit) }, 200, 45);
};

const traidePlans: Handler = async (req, url, env) => {
  if (!traideGuard(req)) return forbidden();
  const limit = clampInt(url.searchParams.get("limit"), 40, 1, 100);
  const days = clampInt(url.searchParams.get("days"), 3, 1, 14);
  const start = Math.floor(Date.now() / 86400000) - days + 1;
  const filter = `PartitionKey eq 'plan' and day ge ${start} and RowKey ge '${start * 86400}'`;
  const items = dataItems(await tableQuery(env, filter, ["data"]));
  return json({ items: items.slice(0, limit) }, 200, 45);
};

// TrAIde/market.py: public candles, proxied because KuCoin sends no CORS headers.
//
// KuCoin first, because the bot trades there and the chart has to line up with its
// entry/TP/SL. From a Worker that is harder than it was from the App Service: Cloudflare's
// egress IPs are shared, KuCoin answers about half the calls with 429, and Binance refuses
// Cloudflare outright (403). So KuCoin gets a quick retry, OKX (same BTC-USDT symbols, prices
// within a few basis points) is the next source, and the last good chart is kept for
// LAST_GOOD_TTL so a bad moment shows slightly older candles instead of an empty chart.
const INTERVALS: Record<string, [string, string, string]> = {
  // display -> [kucoin type, okx bar, binance interval]
  "15min": ["15min", "15m", "15m"],
  "1hour": ["1hour", "1H", "1h"],
  "4hour": ["4hour", "4H", "4h"],
};
const MAX_CANDLES = 160;
const LAST_GOOD_TTL = 900;

export function validSymbol(symbol: string): boolean {
  const parts = symbol.split("-");
  if (parts.length !== 2) return false;
  const [b, q] = parts;
  return /^[A-Za-z0-9]{1,12}$/.test(b) && /^[A-Za-z0-9]{2,6}$/.test(q);
}

type Candle = { t: number; o: number; h: number; l: number; c: number };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function fromKucoin(symbol: string, ktype: string): Promise<Candle[]> {
  const url = `https://api.kucoin.com/api/v1/market/candles?type=${ktype}&symbol=${symbol}`;
  let res = await fetch(url);
  for (const wait of [250, 750]) {
    if (res.status !== 429) break;
    await sleep(wait);
    res = await fetch(url);
  }
  if (!res.ok) throw new Error(`kucoin ${res.status}`);
  const body = (await res.json()) as { code?: string; msg?: string; data?: string[][] };
  if (!body.data?.length) throw new Error(`kucoin ${body.code} ${body.msg ?? "no data"}`);
  const out: Candle[] = [];
  for (const r of body.data) {
    // KuCoin rows are [time(s), open, CLOSE, high, low, ...], newest first.
    const c = { t: parseInt(r[0], 10), o: +r[1], c: +r[2], h: +r[3], l: +r[4] };
    if ([c.t, c.o, c.c, c.h, c.l].every(Number.isFinite)) out.push(c);
  }
  return out.sort((a, b) => a.t - b.t);
}

async function fromOkx(symbol: string, bar: string): Promise<Candle[]> {
  const res = await fetch(
    `https://www.okx.com/api/v5/market/candles?instId=${symbol}&bar=${bar}&limit=${MAX_CANDLES}`,
  );
  if (!res.ok) throw new Error(`okx ${res.status}`);
  const body = (await res.json()) as { code?: string; msg?: string; data?: string[][] };
  if (!body.data?.length) throw new Error(`okx ${body.code} ${body.msg || "no data"}`);
  const out: Candle[] = [];
  for (const r of body.data) {
    // OKX rows are [time(ms), open, high, low, close, ...], newest first.
    const c = { t: Math.floor(Number(r[0]) / 1000), o: +r[1], h: +r[2], l: +r[3], c: +r[4] };
    if ([c.t, c.o, c.c, c.h, c.l].every(Number.isFinite)) out.push(c);
  }
  return out.sort((a, b) => a.t - b.t);
}

async function fromBinance(symbol: string, biv: string): Promise<Candle[]> {
  const res = await fetch(
    `https://api.binance.com/api/v3/klines?symbol=${symbol.replace("-", "")}&interval=${biv}&limit=${MAX_CANDLES}`,
  );
  if (!res.ok) throw new Error(`binance ${res.status}`);
  const out: Candle[] = [];
  for (const r of (await res.json()) as (string | number)[][]) {
    const c = { t: Math.floor(Number(r[0]) / 1000), o: +r[1], h: +r[2], l: +r[3], c: +r[4] };
    if ([c.t, c.o, c.c, c.h, c.l].every(Number.isFinite)) out.push(c);
  }
  return out.sort((a, b) => a.t - b.t);
}

const traideCandles: Handler = async (req, url, env) => {
  if (!traideGuard(req)) return forbidden();
  const symbol = (url.searchParams.get("symbol") || "").toUpperCase().trim();
  let interval = url.searchParams.get("interval") || "1hour";
  if (!(interval in INTERVALS)) interval = "1hour";
  if (!validSymbol(symbol)) return json({ symbol, interval, candles: [] }, 200, 45);
  const [ktype, okxBar, biv] = INTERVALS[interval];
  const sources: [string, () => Promise<Candle[]>][] = [
    ["kucoin", () => fromKucoin(symbol, ktype)],
    ["okx", () => fromOkx(symbol, okxBar)],
    ["binance", () => fromBinance(symbol, biv)],
  ];
  const lastGoodKey = new Request(`https://candles.internal/${symbol}/${interval}`);
  let candles: Candle[] = [];
  for (const [name, source] of sources) {
    try {
      candles = (await source()).slice(-MAX_CANDLES);
      if (candles.length) break;
    } catch (err) {
      console.log(`candles ${symbol} ${interval}: ${name} failed: ${(err as Error).message}`);
      candles = [];
    }
  }
  if (candles.length) {
    await caches.default.put(lastGoodKey, json({ symbol, interval, candles }, 200, LAST_GOOD_TTL));
    return json({ symbol, interval, candles }, 200, 45);
  }
  const stale = await caches.default.match(lastGoodKey);
  if (stale) {
    console.log(`candles ${symbol} ${interval}: every source failed, serving the last good chart`);
    const body = await stale.text();
    return text(body, 200, { "content-type": "application/json" }, 5);
  }
  // Nothing at all: keep the empty answer out of the edge cache for long.
  return json({ symbol, interval, candles: [] }, 200, 5);
};

// ---------------------------------------------------------------------------
// Routing and the edge cache
// ---------------------------------------------------------------------------

const ROUTES: Record<string, Handler> = {
  "/tomorrownewscontent": tomorrownewsContent,
  "/tomorrownewsindex": tomorrownewsIndex,
  "/aiblogcontent": aiblogContent,
  "/aiblogindex": aiblogIndex,
  "/comicbookcontent": comicbookContent,
  "/comicbookindex": comicbookIndex,
  "/cbimg": cbimg,
  "/get-string": getString,
  "/genbox-channels": genboxChannels,
  "/genbox-video-status": genboxVideoStatus,
  "/genbox-video": genboxMedia("video"),
  "/genbox-audio": genboxMedia("audio"),
  "/ai-open-problem-solver/problems": aiopsProblems,
  "/ai-open-problem-solver/problem-details": aiopsDetails,
  "/ai-open-problem-solver/history": aiopsHistory,
  "/traide/live": traideLive,
  "/traide/equity": traideEquity,
  "/traide/feed": traideFeed,
  "/traide/trades": traideTrades,
  "/traide/plans": traidePlans,
  "/traide/candles": traideCandles,
};

// Routes whose answer depends only on the URL (the Referer guards run before the cache).
// Anything time-relative ("today", "no date") is cached short by its own max-age.
const CACHEABLE = new Set(Object.keys(ROUTES).filter((p) => p !== "/genbox-video" && p !== "/genbox-audio"));

// ---------------------------------------------------------------------------
// The generation trigger (cron)
// ---------------------------------------------------------------------------
// GitHub's scheduler is late and lossy on this account: crons run hours late or not at all.
// So this Worker's cron starts the Generate workflow itself (workflow_dispatch), on the same
// cadence as the `cron:` lines in .github/workflows/generate.yml, which stay as a fallback.
// It decides nothing: the workflow's planner (jobs/needed.py) skips a project whose content is
// already published, and its concurrency group stops two runs of a project overlapping.

// Projects due at a UTC hour. Mirrors generate.yml (and the Logic Apps before it).
export function dueProjects(hour: number): string[] {
  const due = ["tomorrownews"]; // hourly (Scheduler01)
  if (hour % 3 === 0) due.push("aiblog"); // every 3 h from 00 (Scheduler02)
  if (hour % 4 === 1) due.push("comicbook"); // 01, 05, ... 21 (Scheduler04)
  if (hour % 3 === 0) due.push("genbox"); // every 3 h from 00 (Scheduler00)
  due.push("aiops"); // hourly (Scheduler03)
  return due;
}

async function dispatchGenerate(env: Env, hour: number): Promise<void> {
  if (!env.GH_DISPATCH_TOKEN) {
    console.log("generate trigger: GH_DISPATCH_TOKEN is not set, nothing started");
    return;
  }
  const failed: string[] = [];
  for (const project of dueProjects(hour)) {
    const res = await fetch(
      `https://api.github.com/repos/${env.GH_REPO}/actions/workflows/generate.yml/dispatches`,
      {
        method: "POST",
        headers: {
          authorization: `Bearer ${env.GH_DISPATCH_TOKEN}`,
          accept: "application/vnd.github+json",
          "x-github-api-version": "2022-11-28",
          "user-agent": "sandbox-data-worker",
          "content-type": "application/json",
        },
        body: JSON.stringify({ ref: "main", inputs: { project } }),
      },
    );
    const detail = res.ok ? "" : ` ${(await res.text()).slice(0, 200)}`;
    console.log(`generate trigger: ${project} -> ${res.status}${detail}`);
    if (!res.ok) failed.push(`${project} ${res.status}`);
  }
  // Throwing marks the cron run as failed in the Cloudflare dashboard, so a dead token shows.
  if (failed.length) throw new Error(`generate trigger failed: ${failed.join(", ")}`);
}

export default {
  async scheduled(event: ScheduledController, env: Env): Promise<void> {
    await dispatchGenerate(env, new Date(event.scheduledTime).getUTCHours());
  },

  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";
    const handler = ROUTES[path];
    if (!handler) return text("404 Not Found", 404);
    if (req.method !== "GET" && req.method !== "HEAD") return text("Method Not Allowed", 405);

    // Guards first: a cached answer must never be handed to a request the guard refuses.
    const guarded = path.startsWith("/traide/")
      ? !traideGuard(req)
      : ["/tomorrownewscontent", "/tomorrownewsindex", "/aiblogcontent", "/aiblogindex",
        "/comicbookcontent", "/genbox-channels"].includes(path) && !req.headers.get("referer");
    if (guarded) return handler(req, url, env);

    const cache = caches.default;
    const cacheKey = new Request(url.toString(), { method: "GET" });
    if (CACHEABLE.has(path)) {
      const hit = await cache.match(cacheKey);
      if (hit) return hit;
    }
    let res: Response;
    try {
      res = await handler(req, url, env);
    } catch (err) {
      console.error(`${path}: ${(err as Error).message}`);
      return path.startsWith("/traide/") || path.startsWith("/ai-open")
        ? json({ error: "upstream error" }, 502, 0)
        : text("Server error", 502, {}, 0);
    }
    if (CACHEABLE.has(path) && res.status === 200) ctx.waitUntil(cache.put(cacheKey, res.clone()));
    return res;
  },
};
