// Serve the assembled site (site/_build, from `make build`) and open it in
// headless Chromium, for the accessibility and keyboard tests.

import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const buildDir = fileURLToPath(new URL("../../_build", import.meta.url));
const axeSource = createRequire(import.meta.url).resolve("axe-core/axe.min.js");

// The success criteria of WCAG 2.2 levels A and AA that axe can test.
const WCAG_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

const CONTENT_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json",
  ".gz": "application/gzip",
  ".png": "image/png",
  ".pdf": "application/pdf",
};

/** Serve site/_build on a free local port; resolves to {url, close()}. */
export async function serveBuild() {
  if (!existsSync(join(buildDir, "index.html"))) {
    throw new Error(`${buildDir} has no index.html; run \`make -C site build\` first`);
  }
  const server = createServer((request, response) => {
    const path = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    const file = join(buildDir, normalize(path.endsWith("/") ? `${path}index.html` : path));
    if (!file.startsWith(buildDir) || !existsSync(file) || !statSync(file).isFile()) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, { "content-type": CONTENT_TYPES[extname(file)] ?? "text/plain" });
    createReadStream(file).pipe(response);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${server.address().port}/`,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}

export async function launch() {
  return chromium.launch();
}

/**
 * Open the page at `width` px and wait, without scrolling, until every
 * interactive section has rendered: a reader navigating by screen reader
 * does not necessarily scroll the viewport.
 */
export async function openPage(browser, url, { width = 1280, height = 900 } = {}) {
  const page = await browser.newPage({ viewport: { width, height } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error));
  await page.goto(url);
  for (const selector of ["#ft-tracks .stack", "#pg-chart svg", "#cond-view .stack", "#rec-view .stack"]) {
    await page.waitForSelector(selector, { timeout: 30_000 });
  }
  page.errors = errors;
  return page;
}

/** axe-core's WCAG 2.2 A/AA violations within `selector`, summarized for assertion messages. */
export async function axeViolations(page, selector = "html") {
  await page.addScriptTag({ path: axeSource });
  const violations = await page.evaluate(
    async ({ selector, tags }) => {
      const result = await window.axe.run(selector, { runOnly: { type: "tag", values: tags } });
      return result.violations;
    },
    { selector, tags: WCAG_TAGS },
  );
  return violations.map(
    (v) => `${v.id} (${v.impact}): ${v.help}\n    ${v.nodes.map((n) => n.target.join(" ")).join("\n    ")}`,
  );
}
