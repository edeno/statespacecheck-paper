// The build-time prerender fills every reported value and DOI link in the page
// from the committed manifest. Run with `node --test site/tests/`.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const siteDir = fileURLToPath(new URL("..", import.meta.url));
const manifestPath = join(siteDir, "data", "manifest.json");
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));

function prerenderedPage() {
  const dir = mkdtempSync(join(tmpdir(), "prerender-"));
  try {
    const page = join(dir, "index.html");
    copyFileSync(join(siteDir, "index.html"), page);
    execFileSync(process.execPath, [join(siteDir, "tools", "prerender.mjs"), page, manifestPath]);
    return readFileSync(page, "utf8");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test("the decoder-input link points at the Figure-4 input file's DOI", () => {
  const html = prerenderedPage();
  const doi = manifest.macros.RecordingInputsDOI;
  assert.match(doi, /^10\.5281\/zenodo\.\d+$/);
  const link = html.match(/<a [^>]*data-doi-macro="RecordingInputsDOI"[^>]*>/);
  assert.ok(link, "the page has a link to the Figure-4 input file");
  assert.ok(link[0].includes(`href="https://doi.org/${doi}"`), link[0]);
});

test("no reported-value placeholder is left empty", () => {
  const html = prerenderedPage();
  assert.equal(html.match(/<span data-macro="[^"]*"[^>]*><\/span>/g), null);
});
