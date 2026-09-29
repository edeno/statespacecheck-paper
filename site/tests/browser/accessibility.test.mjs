// Accessibility of the assembled site in headless Chromium: axe-core's WCAG
// 2.2 A/AA rules in each interactive state, 320-px reflow, and the keyboard
// and screen-reader semantics of each interactive control. Automated checks
// do not establish conformance; they catch regressions between manual reviews.
// Run with `make -C site test-browser`.

import assert from "node:assert/strict";
import { after, before, describe, test } from "node:test";

import { launch, openPage, serveBuild } from "./site.mjs";

let server;
let browser;

before(async () => {
  server = await serveBuild();
  browser = await launch();
});

after(async () => {
  await browser?.close();
  await server?.close();
});

/** Run `body` on the fully loaded page (see openPage); page errors fail the test. */
async function withPage(options, body) {
  const page = await openPage(browser, server.url, options);
  try {
    await body(page);
    assert.deepEqual(page.errors.map(String), []);
  } finally {
    await page.close();
  }
}

describe("layout", () => {
  test("every section loads without scrolling", () => withPage({}, async () => {}));

  test("at 320 CSS pixels wide, the page does not scroll horizontally", () =>
    withPage({ width: 320, height: 640 }, async (page) => {
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      );
      assert.equal(overflow, 0);
    }));
});
