// Chart marks reach WCAG 1.4.11's 3:1 non-text contrast against the page's
// light surfaces. The paper's palette tokens stay as the figures define them
// (tests/test_site_export.py); marks too faint on these surfaces use a darker
// `-ink` shade of the same hue, and the page draws with that shade.

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
const INKS = Object.keys(tokens).filter((name) => name.endsWith("-ink"));
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

function hue(hex) {
  const [r, g, b] = channels(hex);
  const max = Math.max(r, g, b);
  const span = max - Math.min(r, g, b);
  const h = max === r ? ((g - b) / span) % 6 : max === g ? (b - r) / span + 2 : (r - g) / span + 4;
  return (h * 60 + 360) % 360;
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

test("each ink is a shade of its palette color's hue", () => {
  assert.ok(INKS.length > 0);
  for (const ink of INKS) {
    const base = ink.slice(0, -"-ink".length);
    assert.ok(tokens[base], `${ink} has no palette color ${base}`);
    const difference = Math.abs(hue(tokens[ink]) - hue(tokens[base]));
    assert.ok(Math.min(difference, 360 - difference) < 1, `${ink} vs ${base}`);
  }
});

test("the page draws with the inks, not the palette colors they darken", () => {
  const uses = [];
  for (const [file, source] of pageSources) {
    for (const ink of INKS) {
      const base = ink.slice(0, -"-ink".length);
      if (new RegExp(`${base}(?![\\w-])`).test(source)) uses.push(`${file}: ${base}`);
    }
  }
  assert.deepEqual(uses, []);
});
