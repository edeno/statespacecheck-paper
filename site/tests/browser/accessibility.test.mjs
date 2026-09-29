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

describe("charts", () => {
  test("each chart is an image whose text alternative describes its current state", () =>
    withPage({}, async (page) => {
      const charts = await page.$$eval("svg.dist-chart", (svgs) =>
        svgs.map((svg) => ({
          role: svg.getAttribute("role"),
          label: svg.getAttribute("aria-label") ?? "",
          controls: svg.querySelectorAll("[tabindex], [role]").length,
        })),
      );
      assert.ok(charts.length >= 8);
      for (const { role, label, controls } of charts) {
        assert.equal(role, "img");
        assert.match(label, /: .*(peaks at|is flat)/, label);
        assert.equal(controls, 0, label);
      }
    }));

  test("the playground's firing cell is a native radio group operated by the arrow keys", () =>
    withPage({}, async (page) => {
      assert.ok((await page.locator("#pg-chart fieldset input[type=radio]").count()) > 1);
      const chart = page.locator("#pg-chart svg");
      const before = await chart.getAttribute("aria-label");
      await page.locator("#pg-chart input[type=radio]:checked").focus();
      await page.keyboard.press("ArrowRight");
      assert.notEqual(await chart.getAttribute("aria-label"), before);
      assert.equal(await page.evaluate(() => document.activeElement.checked), true);
      assert.equal(await page.locator("#pg-chart .cell-hit.focused").count(), 1);
      // The sparse epoch replaces the cell set, and with it the radio group.
      await page.click("[data-ensemble=sparse_epoch]");
      assert.equal(await page.locator("#pg-chart input[type=radio]:checked").count(), 1);
    }));
});

describe("tables", () => {
  test("every table has a caption, and each value a row and a column header", () =>
    withPage({}, async (page) => {
      const tables = await page.$$eval(".compare-table", (tables) =>
        tables.map((table) => ({
          caption: Boolean(table.caption?.textContent.trim()),
          emptyHeaders: [...table.querySelectorAll("th")].filter((th) => !th.textContent.trim()).length,
          columnHeaders: [...table.tHead.rows[0].cells].every((th) => th.scope === "col"),
          rowHeaders: [...table.tBodies[0].rows].every(
            (row) => row.cells[0].tagName === "TH" && row.cells[0].scope === "row",
          ),
        })),
      );
      assert.ok(tables.length >= 2);
      for (const table of tables) {
        assert.deepEqual(table, { caption: true, emptyHeaders: 0, columnHeaders: true, rowHeaders: true });
      }
    }));
});
