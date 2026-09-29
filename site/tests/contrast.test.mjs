// Chart marks reach WCAG 1.4.11's 3:1 non-text contrast against the page's
// light surfaces.

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const siteDir = fileURLToPath(new URL("..", import.meta.url));
const css = readFileSync(join(siteDir, "css", "style.css"), "utf8");
const root = css.match(/:root\s*{([^}]*)}/)[1];
const tokens = Object.fromEntries(
  [...root.matchAll(/(--[\w-]+):\s*(#[0-9a-fA-F]{6})\s*;/g)].map(([, name, hex]) => [name, hex]),
);

const pageFiles = [
  "index.html",
  ...readdirSync(join(siteDir, "js"))
    .filter((name) => name.endsWith(".js"))
    .map((name) => join("js", name)),
];
const pageSources = pageFiles.map((file) => [file, readFileSync(join(siteDir, file), "utf8")]);

// The surfaces charts and tracks are drawn on: page, panel, and shaded window.
const SURFACES = ["--surface", "--surface-muted", "--surface-sunken"];
// Colors the page draws with that carry no information: gridlines, borders,
// and the faint fields of the cells that did not fire in the filter explainer,
// context beside the highlighted field.
const DECORATION = ["--grid", "--border", "--field-muted"];
// Every other color the page's markup and scripts use marks something a reader needs.
const used = new Set(pageSources.flatMap(([, source]) => source.match(/--[\w-]+/g) ?? []));
const MARKS = Object.keys(tokens).filter(
  (name) => used.has(name) && !SURFACES.includes(name) && !DECORATION.includes(name),
);

function channels(hex) {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
}

function luminance(hex) {
  const [r, g, b] = channels(hex).map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a, b) {
  const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (high + 0.05) / (low + 0.05);
}

test("every chart mark has 3:1 contrast against each surface", () => {
  assert.ok(MARKS.includes("--predictive"), MARKS.join(", "));
  const failures = [];
  for (const mark of MARKS) {
    for (const surface of SURFACES) {
      const ratio = contrast(tokens[mark], tokens[surface]);
      if (!(ratio >= 3)) failures.push(`${mark} on ${surface}: ${ratio.toFixed(2)}:1`);
    }
  }
  assert.deepEqual(failures, []);
});
