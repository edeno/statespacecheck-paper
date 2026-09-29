// Accessibility of the assembled site in headless Chromium: axe-core's WCAG
// 2.2 A/AA rules in each interactive state, 320-px reflow, and the keyboard
// and screen-reader semantics of each interactive control. Automated checks
// do not establish conformance; they catch regressions between manual reviews.
// Run with `make -C site test-browser`.

import assert from "node:assert/strict";
import { after, before, describe, test } from "node:test";

import { axeViolations, launch, openPage, serveBuild } from "./site.mjs";

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

/** Wait until the condition panel shows condition `id`. */
async function conditionShown(page, id) {
  await page.waitForFunction((id) => {
    const view = document.querySelector("#cond-view");
    return view.getAttribute("aria-labelledby") === `cond-tab-${id}` && view.querySelector(".stack");
  }, id);
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

describe("players", () => {
  test("track stacks are sliders whose value text names the selection", () =>
    withPage({}, async (page) => {
      for (const selector of ["#ft-tracks .stack", "#cond-view .stack", "#rec-view .stack"]) {
        const stack = page.locator(selector);
        assert.equal(await stack.getAttribute("role"), "slider", selector);
        await stack.focus();
        const before = await stack.getAttribute("aria-valuetext");
        await page.keyboard.press("ArrowRight");
        const after = await stack.getAttribute("aria-valuetext");
        assert.ok(before && after, selector);
        assert.notEqual(after, before, selector);
        // Spikes in one time bin share a time, so the value itself may stay.
        const value = Number(await stack.getAttribute("aria-valuenow"));
        const min = Number(await stack.getAttribute("aria-valuemin"));
        const max = Number(await stack.getAttribute("aria-valuemax"));
        assert.ok(min <= value && value <= max, `${selector}: ${min} <= ${value} <= ${max}`);
      }
    }));

  test("hovering a focused track stack does not change its spoken value", () =>
    withPage({}, async (page) => {
      const stack = page.locator("#rec-view .stack");
      await stack.scrollIntoViewIfNeeded();
      const box = await stack.boundingBox();
      const y = box.y + box.height / 2;
      await page.mouse.click(box.x + box.width * 0.6, y);
      assert.equal(await page.evaluate(() => document.activeElement.classList.contains("stack")), true);
      const pressed = await stack.getAttribute("aria-valuetext");
      for (const fraction of [0.3, 0.4, 0.5, 0.7, 0.9]) {
        await page.mouse.move(box.x + box.width * fraction, y);
      }
      assert.equal(await stack.getAttribute("aria-valuetext"), pressed);
      // The keys step from the spike under the cursor, where hovering left it.
      await page.keyboard.press("ArrowUp");
      const up = await stack.getAttribute("aria-valuetext");
      assert.notEqual(up, pressed);
      await page.keyboard.press("ArrowDown");
      assert.notEqual(await stack.getAttribute("aria-valuetext"), up);
    }));

  test("dragging along the tracks leaves the spoken value on the spike shown", () =>
    withPage({}, async (page) => {
      const stack = page.locator("#rec-view .stack");
      await stack.scrollIntoViewIfNeeded();
      const box = await stack.boundingBox();
      const y = box.y + box.height / 2;
      await page.mouse.move(box.x + box.width * 0.05, y);
      await page.mouse.down();
      const pressed = await stack.getAttribute("aria-valuetext");
      await page.mouse.move(box.x + box.width * 0.4, y, { steps: 5 });
      // Released outside the tracks: the pointer is captured.
      await page.mouse.move(box.x + box.width * 0.7, box.y - 40, { steps: 5 });
      await page.mouse.up();
      const shown = await page.locator("#rec-view .detail h3").textContent();
      const spoken = await stack.getAttribute("aria-valuetext");
      assert.notEqual(spoken, pressed);
      assert.ok(spoken.startsWith(shown), `${spoken} / ${shown}`);
    }));

  test("each player lists every spike in a table", () =>
    withPage({}, async (page) => {
      for (const view of ["#cond-view", "#rec-view"]) {
        const details = page.locator(`${view} details.spike-table`);
        const count = Number((await details.locator("summary").textContent()).match(/\d+/)[0]);
        await details.locator("summary").click();
        await details.locator("tbody tr").first().waitFor();
        assert.equal(await details.locator("tbody tr").count(), count, view);
        assert.ok(count > 10, view);
      }
    }));

  test("the recording model switch preserves the spike and changes the plotted prediction", () =>
    withPage({}, async (page) => {
      const view = page.locator("#rec-view");
      const charts = view.locator(".detail svg.dist-chart");
      const state = async () => ({
        spike: await view.locator(".detail h3").textContent(),
        spikeChart: await charts.first().getAttribute("aria-label"),
        fieldChart: await charts.last().getAttribute("aria-label"),
        prediction: await view.locator(".stack canvas").first().evaluate((c) => c.toDataURL()),
        // Every track label but the prediction's, which names the model.
        otherLabels: (await view.locator(".stack .track-label").allTextContents()).slice(1),
      });
      const radios = view.locator(".model-switch input[type=radio]");
      assert.equal(await radios.count(), 2);
      const [first, second] = [radios.first(), radios.last()];
      assert.equal(await first.isChecked(), true);
      const before = await state();

      await second.check();
      const switched = await state();
      assert.equal(switched.spike, before.spike);
      assert.equal(switched.fieldChart, before.fieldChart);
      assert.deepEqual(switched.otherLabels, before.otherLabels);
      assert.notEqual(switched.spikeChart, before.spikeChart);
      assert.notEqual(switched.prediction, before.prediction);
      const spoken = await view.locator(".stack").getAttribute("aria-valuetext");
      assert.match(spoken, /Continuous.Fragmented model/);

      await first.check();
      assert.deepEqual(await state(), before);
      await first.focus();
      await page.keyboard.press("ArrowRight");
      assert.equal(await second.isChecked(), true);
      assert.equal(await view.locator(".detail h3").textContent(), before.spike);
    }));

  test("the selected spike shows its unit's exported place field", () =>
    withPage({}, async (page) => {
      const view = page.locator("#rec-view");
      const heading = () => view.locator(".detail .row-label").textContent();
      const fieldChart = () => view.locator(".detail svg.dist-chart").last().getAttribute("aria-label");
      const unit = Number((await view.locator(".detail h3").textContent()).match(/unit (\d+)/)[1]);
      assert.match(await heading(), new RegExp(`unit ${unit} \\(`));
      assert.match(await heading(), /peak [\d.e+-]+ expected spikes per \d+-ms time bin/);
      assert.match(await fieldChart(), new RegExp(`unit ${unit} place field peaks at`));

      const exported = await page.evaluate(async () => (await fetch("data/recording.json")).json());
      const nUnits = exported.cell_rank.length;
      assert.equal(exported.place_fields.row_max.length, nUnits);
      // One row of the position grid per unit.
      assert.equal(atob(exported.place_fields.rows).length, nUnits * exported.position_bins.length);

      // Jump to the first or last spike, whichever is from another unit.
      const cells = exported.models.continuous.events.cell;
      const [key, other] = cells[0] + 1 !== unit ? ["Home", cells[0] + 1] : ["End", cells.at(-1) + 1];
      assert.notEqual(other, unit);
      await view.locator(".stack").focus();
      await page.keyboard.press(key);
      assert.match(await heading(), new RegExp(`unit ${other} \\(`));
      assert.match(await fieldChart(), new RegExp(`unit ${other} place field peaks at`));
    }));
});

describe("recording explorer", () => {
  /** Choose the square of `metric` whose count is closest to `target` by clicking it. */
  async function clickSquare(page, metricIndex, target) {
    const canvas = page.locator("#rec-explorer canvas").nth(metricIndex);
    await canvas.scrollIntoViewIfNeeded();
    const overview = await page.evaluate(async () => (await fetch("data/recording_explorer.json")).json());
    const name = Object.keys(overview.metrics)[metricIndex];
    const { counts } = overview.metrics[name];
    const square = counts.reduce(
      (best, count, i) => (count && Math.abs(count - target) < Math.abs(counts[best] - target) ? i : best),
      counts.findIndex(Boolean),
    );
    const plot = JSON.parse(await canvas.getAttribute("data-plot"));
    const bounds = await canvas.boundingBox();
    const scale = bounds.width / plot.canvas;
    const cell = plot.size / plot.grid;
    const [x, y] = [Math.floor(square / plot.grid), square % plot.grid];
    await page.mouse.click(
      bounds.x + (plot.left + (x + 0.5) * cell) * scale,
      bounds.y + (plot.top + (plot.grid - y - 0.5) * cell) * scale,
    );
    return { name, square, count: counts[square] };
  }

  const selectedId = (page) => Number(new URL(page.url()).searchParams.get("recording_event"));

  async function spikeShown(page) {
    await page.waitForFunction(() => new URL(location.href).searchParams.has("recording_event"));
    await page.locator("#rec-explorer a[href='#rec-view']").waitFor();
  }

  /** The session index, decoded as the page decodes it. */
  const sessionIndex = (page) =>
    page.evaluate(async () => {
      const { loadContainer } = await import("./js/container.js");
      const { arrays } = await loadContainer("data/explorer/index.bin.gz");
      return Object.fromEntries(Object.entries(arrays).map(([k, v]) => [k, Array.from(v)]));
    });

  test("the large files load only when a reader browses", () =>
    withPage({}, async (page) => {
      await page.locator("#rec-explorer canvas").first().waitFor();
      const fetched = () =>
        page.evaluate(() =>
          performance.getEntriesByType("resource").map((e) => e.name).filter((n) => /explorer\//.test(n)),
        );
      assert.deepEqual(await fetched(), []);
      await clickSquare(page, 0, 10);
      await spikeShown(page);
      const names = await fetched();
      assert.equal(names.filter((n) => n.endsWith("index.bin.gz")).length, 1);
      // One window: two consecutive blocks.
      assert.equal(names.filter((n) => /blocks\/\d{4}\.bin\.gz$/.test(n)).length, 2);
    }));

  test("a square opens one of its spikes, the one the player then shows", () =>
    withPage({}, async (page) => {
      const { name, square, count } = await clickSquare(page, 0, 10);
      await spikeShown(page);
      const index = await sessionIndex(page);
      const id = selectedId(page);
      assert.equal(index[`square_${name}`][id], square);
      assert.equal(index[`square_${name}`].filter((s) => s === square).length, count);
      const unit = index.cell[id] + 1;
      assert.match(await page.locator("#rec-view .detail h3").textContent(), new RegExp(`unit ${unit}$`));
      const status = await page.locator("#rec-explorer [aria-live]").textContent();
      assert.match(status, new RegExp(`of ${count} in this square: unit ${unit},`));
    }));

  test("the controls reach a spike by keyboard and follow the player", () =>
    withPage({}, async (page) => {
      await clickSquare(page, 1, 50);
      await spikeShown(page);
      const first = selectedId(page);
      const spike = page.locator("#rec-explorer input[type=range]").nth(2);
      await spike.focus();
      await page.keyboard.press("ArrowRight");
      await page.waitForFunction((old) => Number(new URL(location.href).searchParams.get("recording_event")) !== old, first);
      const second = selectedId(page);
      assert.match(await spike.getAttribute("aria-valuetext"), /Spike \d+ of \d+ in this square/);
      // Stepping in the player moves the explorer's selection with it.
      await page.locator("#rec-view .stack").focus();
      await page.keyboard.press("ArrowRight");
      await page.waitForFunction((old) => Number(new URL(location.href).searchParams.get("recording_event")) !== old, second);
      const third = selectedId(page);
      assert.equal(await page.locator("#rec-explorer a", { hasText: "Link to the selected spike" }).getAttribute("href"), page.url());
      const index = await sessionIndex(page);
      const square = index.square_predictive_pvalue[third];
      const position = index.square_predictive_pvalue.slice(0, third + 1).filter((s) => s === square).length;
      assert.equal(await spike.inputValue(), String(position));
    }));

  test("a link reopens its spike, and the model switch keeps it", () =>
    withPage({}, async (page) => {
      await clickSquare(page, 2, 30);
      await spikeShown(page);
      const id = selectedId(page);
      const detail = await page.locator("#rec-view .detail h3").textContent();
      await page.goto(page.url());
      await page.waitForFunction((target) =>
        document.querySelector("#rec-explorer a[href='#rec-view']")?.hidden === false &&
        new URL(location.href).searchParams.get("recording_event") === String(target), id);
      assert.equal(await page.locator("#rec-view .detail h3").textContent(), detail);
      await page.locator("#rec-view .model-switch input[type=radio]").last().check();
      assert.equal(await page.locator("#rec-view .detail h3").textContent(), detail);
    }));

  test("the explorer passes axe and reflows at 320 CSS pixels", () =>
    withPage({ width: 320, height: 640 }, async (page) => {
      await clickSquare(page, 0, 10);
      await spikeShown(page);
      assert.deepEqual(await axeViolations(page, "#real-data"), []);
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      );
      assert.equal(overflow, 0);
    }));
});

describe("condition tabs", () => {
  test("the arrow keys move focus between tabs and Enter selects one", () =>
    withPage({}, async (page) => {
      const selected = page.locator("#cond-tabs [aria-selected=true]");
      const first = await selected.getAttribute("data-condition");
      await selected.focus();
      await page.keyboard.press("ArrowRight");
      const focused = await page.evaluate(() => document.activeElement.dataset.condition);
      assert.notEqual(focused, first);
      assert.equal(await page.locator(`#cond-tab-${focused}`).getAttribute("aria-selected"), "false");
      await page.keyboard.press("Enter");
      await conditionShown(page, focused);
      assert.equal(await page.locator(`#cond-tab-${focused}`).getAttribute("aria-selected"), "true");
    }));

  test("Tab from the selected tab enters the panel at its start", () =>
    withPage({}, async (page) => {
      await page.locator("#cond-tabs [aria-selected=true]").focus();
      await page.keyboard.press("Tab");
      assert.equal(await page.evaluate(() => document.activeElement.id), "cond-view");
      const description = await page.locator("#cond-view").getAttribute("aria-describedby");
      assert.ok((await page.locator(`#${description}`).textContent()).length > 20);
    }));

  test("only a reader's own choice of condition enters the URL", () =>
    withPage({}, async (page) => {
      assert.equal(new URL(page.url()).search, "");
      const other = page.locator("#cond-tabs [aria-selected=false]").first();
      const id = await other.getAttribute("data-condition");
      await other.click();
      await conditionShown(page, id);
      assert.equal(new URL(page.url()).searchParams.get("condition"), id);
    }));
});

describe("navigation and states", () => {
  test("a skip link is the first tab stop and moves focus to the main content", () =>
    withPage({}, async (page) => {
      await page.keyboard.press("Tab");
      const link = await page.evaluate(() => ({
        text: document.activeElement.textContent.trim(),
        visible: document.activeElement.getBoundingClientRect().top >= 0,
      }));
      assert.deepEqual(link, { text: "Skip to content", visible: true });
      await page.keyboard.press("Enter");
      assert.equal(await page.evaluate(() => document.activeElement.tagName), "MAIN");
    }));

  test("nothing without a destination is presented as a link", () =>
    withPage({}, async (page) => {
      assert.equal(await page.locator("[role=link]:not([href]), a:not([href])").count(), 0);
    }));

  test("loading and error messages are status messages", () =>
    withPage({}, async (page) => {
      let fail;
      const failed = new Promise((resolve) => {
        fail = resolve;
      });
      await page.route("**/data/condition_*.json", async (route) => {
        await failed;
        await route.abort();
      });
      const other = page.locator("#cond-tabs [aria-selected=false]").first();
      await other.click();
      const loading = page.locator("#cond-view .loading");
      await loading.waitFor();
      assert.equal(await loading.getAttribute("role"), "status");
      fail();
      const error = page.locator("#cond-view .error");
      await error.waitFor();
      assert.equal(await error.getAttribute("role"), "alert");
    }));
});

describe("axe-core WCAG 2.2 A/AA rules", () => {
  test("the fully loaded page, with both spike tables open", () =>
    withPage({}, async (page) => {
      for (const view of ["#cond-view", "#rec-view"]) {
        await page.click(`${view} details.spike-table summary`);
        await page.locator(`${view} details.spike-table tbody tr`).first().waitFor();
      }
      assert.deepEqual(await axeViolations(page), []);
    }));

  test("the fully loaded page at 320 CSS pixels wide", () =>
    withPage({ width: 320, height: 640 }, async (page) => {
      assert.deepEqual(await axeViolations(page), []);
    }));

  test("the filter explainer while playing and at its last step", () =>
    withPage({}, async (page) => {
      await page.click("#ft-play");
      assert.deepEqual(await axeViolations(page, "#filter"), [], "playing");
      await page.click("#ft-play");
      await page.locator("#ft-tracks .stack").focus();
      await page.keyboard.press("End");
      assert.deepEqual(await axeViolations(page, "#filter"), [], "last step");
    }));

  test("every simulated condition", () =>
    withPage({}, async (page) => {
      const ids = await page.$$eval("#cond-tabs [role=tab]", (tabs) =>
        tabs.map((tab) => tab.dataset.condition),
      );
      assert.ok(ids.length > 1);
      for (const id of ids) {
        await page.click(`#cond-tab-${id}`);
        await conditionShown(page, id);
        assert.deepEqual(await axeViolations(page, "#simulation"), [], id);
      }
    }));

  test("every playground example and cell set", () =>
    withPage({}, async (page) => {
      for (const button of await page.$$("#playground [data-preset], #playground [data-ensemble]")) {
        await button.click();
        const name = await button.textContent();
        assert.deepEqual(await axeViolations(page, "#playground"), [], name);
      }
    }));

  test("both recording model views", () =>
    withPage({}, async (page) => {
      for (const radio of await page.$$("#rec-view .model-switch input[type=radio]")) {
        await radio.check();
        assert.deepEqual(await axeViolations(page, "#real-data"), []);
      }
    }));
});

describe("text layout", () => {
  test("at 320 CSS pixels wide, chart text stays within its chart", () =>
    withPage({ width: 320, height: 640 }, async (page) => {
      const overflows = await page.$$eval("svg.dist-chart", (svgs) =>
        svgs.flatMap((svg) => {
          const right = svg.getBoundingClientRect().right;
          return [...svg.querySelectorAll("text")]
            .filter((text) => text.getBoundingClientRect().right > right + 0.5)
            .map((text) => text.textContent);
        }),
      );
      assert.deepEqual(overflows, []);
    }));

  test("a reported value in a legend sits against the text after it", () =>
    withPage({}, async (page) => {
      const gaps = await page.$$eval(".legend [data-macro]", (values) =>
        values.map((value) => {
          const range = document.createRange();
          range.setStart(value.nextSibling, 0);
          range.setEnd(value.nextSibling, 1);
          return range.getBoundingClientRect().left - value.getBoundingClientRect().right;
        }),
      );
      assert.ok(gaps.length > 0);
      for (const gap of gaps) assert.ok(Math.abs(gap) < 1, `gap ${gap}`);
    }));
});
