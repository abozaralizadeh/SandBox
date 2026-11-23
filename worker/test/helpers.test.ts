// Unit tests for the Worker's pure helpers. The endpoint behaviour itself was checked against
// the Flask app URL by URL (same bodies, status codes and Timestamp/Edition-Missing/Arc-Id
// headers); these pin down the small conversions that comparison depends on.
import { describe, expect, it } from "vitest";
import { derivedName, dueProjects, parseWall, problemSlug, splitArc, stamp, tnBase, validSymbol } from "../src/index";

describe("parseWall (Python's fromisoformat, as the pages use it)", () => {
  it("reads the local ISO time the pages send", () => {
    expect(parseWall("2026-11-23T14:05:09")).toEqual({ y: 2026, m: 11, d: 23, h: 14, mi: 5, s: 9 });
    expect(parseWall("2026-11-23")).toEqual({ y: 2026, m: 11, d: 23, h: 0, mi: 0, s: 0 });
    expect(parseWall("2026-11-23T14:05:09.123Z")?.h).toBe(14);
  });
  it("returns null where Python would raise", () => {
    expect(parseWall("bogus")).toBeNull();
    expect(parseWall("2026-13-01")).toBeNull();
    expect(parseWall("")).toBeNull();
    expect(parseWall(null)).toBeNull();
  });
});

describe("TomorrowNews row keys", () => {
  it("is hourly before 2025-01-25 and daily after", () => {
    expect(tnBase(parseWall("2024-06-01T13:00:00"))).toBe("20240601_13");
    expect(tnBase(parseWall("2025-01-25T13:00:00"))).toBe("20250125_00");
  });
  it("formats the Timestamp header like Flask did", () => {
    expect(stamp("20240601_13")).toBe("2024-06-01 13:00:00");
    expect(stamp("20261123")).toBe("2026-11-23 00:00:00");
  });
});

describe("ComicBook", () => {
  it("splits the arc id off a published page", () => {
    expect(splitArc("<!--arc:arc_1-->\n<p>x</p>")).toEqual(["arc_1", "<p>x</p>"]);
    expect(splitArc("<p>no marker</p>")).toEqual(["", "<p>no marker</p>"]);
  });
  it("names WebP variants like imageproxy._derived_name", () => {
    expect(derivedName("a/b.png", null, 92)).toBe("a/b_q92.webp");
    expect(derivedName("a/b.png", 800, 80)).toBe("a/b_w800_q80.webp");
  });
});

describe("AIOPS problem slug", () => {
  it("matches sitepub: sha1 of the lower-cased, space-collapsed name", async () => {
    // python3 -c "import hashlib; print(hashlib.sha1(b'riemann hypothesis').hexdigest())"
    expect(await problemSlug("  Riemann   Hypothesis ")).toBe("d6c9eef23eff0849ba8388f1751d8d2f12844cc6");
  });
});

describe("candles symbol guard", () => {
  it("accepts BASE-QUOTE only", () => {
    expect(validSymbol("BTC-USDT")).toBe(true);
    expect(validSymbol("BTC")).toBe(false);
    expect(validSymbol("BTC-USDT-X")).toBe(false);
    expect(validSymbol("../etc-x")).toBe(false);
  });
});

describe("generation trigger cadence", () => {
  it("matches the cron lines in generate.yml", () => {
    expect(dueProjects(0)).toEqual(["tomorrownews", "aiblog", "genbox", "aiops"]);
    expect(dueProjects(1)).toEqual(["tomorrownews", "comicbook", "aiops"]);
    expect(dueProjects(2)).toEqual(["tomorrownews", "aiops"]);
    expect(dueProjects(21)).toEqual(["tomorrownews", "aiblog", "comicbook", "genbox", "aiops"]);
  });
  it("starts every project at least once a day", () => {
    const seen = new Set(Array.from({ length: 24 }, (_, h) => dueProjects(h)).flat());
    expect([...seen].sort()).toEqual(["aiblog", "aiops", "comicbook", "genbox", "tomorrownews"]);
  });
});
