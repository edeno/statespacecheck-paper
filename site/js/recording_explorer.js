// Figure-4 session explorer: plots comparing the two decoders over every spike
// in the session, linked to the recording player. The overview is committed;
// the spike index and the two-second recording blocks are fetched when a reader
// first chooses a square, from files the site build extracts from the archive
// the overview names (statespacecheck_paper.site_explorer_export). No decoder
// runs in the browser: every value comes from the paper's fingerprinted decode.

import { cssVar } from "./charts.js";
import { loadContainer } from "./container.js";
import { loadJSON, METRICS } from "./data.js";
import { renderRecording } from "./players.js";

// Canvas size and plot area, in CSS pixels.
const SIZE = 292;
const PLOT = { left: 42, top: 12, size: 230 };
const AXIS_LABEL = {
  hpd_overlap: "HPD overlap",
  predictive_pvalue: "−ln p",
  kl_divergence: "KL divergence (nats)",
};
// Keyboard steps on a range input each fire "change"; act once they pause.
const SETTLE_MS = 300;
const BLOCK_CACHE_SIZE = 6;
const TICKS_PER_SECOND = 10_000;

function hexToRgb(hex) {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

/**
 * Square colors from the metric's color (one spike) to the text color (the
 * fullest square), on a log scale. Both ends have 3:1 contrast with the page,
 * and mixing two such colors keeps it, so sparse squares stay visible.
 */
function densityColor(low, high, fraction) {
  const [a, b] = [hexToRgb(low), hexToRgb(high)];
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * fraction)).join(",")})`;
}

/** Value on the comparison axes: −ln p for the p-value. */
function plotted(metricName, value) {
  return metricName === "predictive_pvalue" ? -Math.log(value) : value;
}

function drawChart(canvas, overview, metric, selectedSquare, highlight) {
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(SIZE * ratio)) {
    canvas.width = Math.round(SIZE * ratio);
    canvas.height = Math.round(SIZE * ratio);
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const grid = overview.grid_size;
  const { max, counts } = overview.metrics[metric.name];
  const { left, top, size } = PLOT;
  const cell = size / grid;
  const peak = Math.max(...counts);
  const [low, high] = [cssVar(metric.color), cssVar("--text")];
  context.fillStyle = cssVar("--surface");
  context.fillRect(0, 0, SIZE, SIZE);
  for (let x = 0; x < grid; x += 1) {
    for (let y = 0; y < grid; y += 1) {
      const count = counts[x * grid + y];
      if (!count) continue;
      const fraction = peak > 1 ? Math.log(count) / Math.log(peak) : 0;
      context.fillStyle = densityColor(low, high, fraction);
      context.fillRect(left + x * cell, top + (grid - 1 - y) * cell, cell + 0.3, cell + 0.3);
    }
  }
  context.save();
  context.beginPath();
  context.rect(left, top, size, size);
  context.clip();
  context.strokeStyle = cssVar("--threshold");
  context.lineWidth = 1.2;
  context.setLineDash([5, 4]);
  context.beginPath();
  context.moveTo(left, top + size);
  context.lineTo(left + size, top);
  context.stroke();
  const rule = overview.flag_rules[metric.name];
  if (rule) {
    const coord = (plotted(metric.name, rule.threshold) / max) * size;
    context.setLineDash([2, 3]);
    context.beginPath();
    context.moveTo(left + coord, top);
    context.lineTo(left + coord, top + size);
    context.moveTo(left, top + size - coord);
    context.lineTo(left + size, top + size - coord);
    context.stroke();
  }
  context.setLineDash([]);
  if (selectedSquare !== null) {
    const x = Math.floor(selectedSquare / grid);
    const y = selectedSquare % grid;
    context.strokeStyle = cssVar("--link");
    context.lineWidth = 2.5;
    context.strokeRect(left + x * cell, top + (grid - 1 - y) * cell, cell, cell);
  }
  if (highlight) {
    const x = left + (Math.min(highlight[0], max) / max) * size;
    const y = top + size - (Math.min(highlight[1], max) / max) * size;
    context.beginPath();
    context.arc(x, y, 5, 0, 2 * Math.PI);
    context.fillStyle = cssVar("--surface");
    context.fill();
    context.strokeStyle = cssVar("--link");
    context.lineWidth = 2;
    context.stroke();
  }
  context.restore();
  context.strokeStyle = cssVar("--text-secondary");
  context.lineWidth = 1;
  context.strokeRect(left, top, size, size);
  context.fillStyle = cssVar("--text-secondary");
  context.font = `12px ${cssVar("--font")}`;
  context.textAlign = "center";
  context.fillText("Continuous", left + size / 2, SIZE - 4);
  context.fillText("0", left, top + size + 15);
  context.fillText(String(max), left + size, top + size + 15);
  context.textAlign = "right";
  context.fillText("0", left - 5, top + size + 4);
  context.fillText(String(max), left - 5, top + 9);
  context.save();
  context.translate(11, top + size / 2);
  context.rotate(-Math.PI / 2);
  context.textAlign = "center";
  context.fillText("Continuous–Fragmented", 0, 0);
  context.restore();
}

function chartElement(metric, overview, onPick) {
  const figure = document.createElement("figure");
  figure.className = "comparison-chart";
  const caption = document.createElement("figcaption");
  caption.textContent = AXIS_LABEL[metric.name];
  const canvas = document.createElement("canvas");
  canvas.style.aspectRatio = "1";
  // The plot area, for pointers and the page's tests.
  canvas.dataset.plot = JSON.stringify({ ...PLOT, canvas: SIZE, grid: overview.grid_size });
  canvas.setAttribute("role", "img");
  canvas.setAttribute(
    "aria-label",
    `${AXIS_LABEL[metric.name]} of every spike: squares count spikes by their value under the Continuous model (horizontal) and the Continuous–Fragmented model (vertical). The table below gives the flag agreement; the controls below choose a square.`,
  );
  canvas.addEventListener("click", (event) => {
    const bounds = canvas.getBoundingClientRect();
    const x = ((event.clientX - bounds.left) / bounds.width) * SIZE;
    const y = ((event.clientY - bounds.top) / bounds.height) * SIZE;
    const { left, top, size } = PLOT;
    if (x < left || x >= left + size || y < top || y >= top + size) return;
    onPick(metric.name, (x - left) / size, 1 - (y - top) / size);
  });
  figure.append(caption, canvas);
  return { figure, canvas };
}

/** The session's flag agreement between the models: the plots' text alternative. */
function agreementTable(overview) {
  const labels = Object.fromEntries(METRICS.map((metric) => [metric.name, metric.label]));
  const table = document.createElement("table");
  table.className = "compare-table";
  const count = (n) => n.toLocaleString("en-US");
  table.innerHTML = `<caption>Spikes flagged across the session, by model</caption><thead><tr><th scope="col">Diagnostic</th><th scope="col">Both models</th><th scope="col">Continuous only</th><th scope="col">Continuous–Fragmented only</th><th scope="col">Neither</th></tr></thead><tbody>${overview.flag_confusions
    .map(
      (row) =>
        `<tr><th scope="row">${labels[row.metric]}</th><td>${count(row.both)}</td><td>${count(row.rescued)}</td><td>${count(row.newly_flagged)}</td><td>${count(row.neither)}</td></tr>`,
    )
    .join("")}</tbody>`;
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.tabIndex = 0;
  wrap.setAttribute("role", "region");
  wrap.setAttribute("aria-label", "Spikes flagged across the session, by model");
  wrap.appendChild(table);
  return wrap;
}

function debounce(action, delay) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => action(...args), delay);
  };
}

function concat(Typed, parts) {
  const out = new Typed(parts.reduce((n, part) => n + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

/**
 * A recording window in recording.json's shape from blocks `blocks` (in order),
 * whose color ranges start at window `windowIndex`. Shared fields (position
 * grid, place fields, per-cell likelihoods, labels) come from `shared`.
 */
function windowPayload(blocks, windowIndex, overview, shared) {
  const [first] = blocks;
  const nTime = blocks.reduce((n, block) => n + block.meta.stop - block.meta.start, 0);
  const round4 = (value) => Math.round(value * 1e4) / 1e4;
  const events = { bin: [], t: [], cell: [], id: [] };
  for (const { meta, arrays } of blocks) {
    const offset = meta.start - first.meta.start;
    const shift = meta.start_time - first.meta.start_time;
    arrays.event_bin.forEach((bin, i) => {
      events.bin.push(bin + offset);
      events.t.push(round4(shift + arrays.event_tick[i] / TICKS_PER_SECOND));
      events.cell.push(arrays.event_cell[i]);
      events.id.push(meta.first_event + i);
    });
  }
  const models = {};
  for (const [id, model] of Object.entries(shared.models)) {
    const values = Object.fromEntries(
      METRICS.map((metric) => [
        metric.name,
        blocks.flatMap(({ arrays }) => Array.from(arrays[`${metric.name}_${id}`])),
      ]),
    );
    const flagged = {};
    first.meta.flag_bits.forEach(([bitModel, metric], bit) => {
      if (bitModel !== id) return;
      flagged[metric] = blocks.flatMap(({ arrays }) =>
        Array.from(arrays.flags, (flags) => ((flags >> bit) & 1) === 1),
      );
    });
    models[id] = {
      label: model.label,
      short_label: model.short_label,
      predictive: {
        rows: concat(Uint8Array, blocks.map(({ arrays }) => arrays[`predictive_${id}`])),
        row_max: blocks.flatMap(({ arrays }) => Array.from(arrays[`predictive_max_${id}`])),
        range: overview.window_ranges[id][windowIndex],
      },
      events: { ...events, ...values, flagged },
    };
  }
  return {
    time: Array.from({ length: nTime }, (_, i) => round4(i * overview.time_step)),
    position_bins: shared.position_bins,
    linear_position: blocks.flatMap(({ arrays }) =>
      Array.from(arrays.position, (value) => (Number.isFinite(value) ? value : null)),
    ),
    cell_likelihoods: shared.cell_likelihoods,
    place_fields: shared.place_fields,
    cell_rank: shared.cell_rank,
    models,
    flag_rules: overview.flag_rules,
    decode_cache_fingerprint: overview.decode_cache_fingerprint,
    diagnostics_fingerprint: overview.diagnostics_fingerprint,
  };
}

/** Show the paper's window, then let readers browse the session from the comparison plots. */
export async function initRecordingExplorer(root, manifest) {
  const view = root.querySelector("#rec-view");
  const explorer = root.querySelector("#rec-explorer");
  const fail = (container, label, error) => {
    const message = document.createElement("p");
    message.className = "error";
    message.setAttribute("role", "alert");
    message.textContent = `Could not load the ${label} (${error.message}).`;
    container.replaceChildren(message);
    console.error(error);
  };

  let player = null;
  let shared = null;
  const paperLoad = loadJSON("data/recording.json").then((payload) => {
    shared = payload;
    player = renderRecording(root, payload, manifest);
  });
  paperLoad.catch((error) => fail(view, "recording", error));
  let overview;
  try {
    overview = await loadJSON("data/recording_explorer.json");
  } catch (error) {
    fail(explorer, "session overview", error);
    return;
  }

  const grid = overview.grid_size;
  const B = overview.block_samples;
  let index = null;
  let indexLoad = null;
  const blockCache = new Map();
  let shown = null; // {window, payload}
  let selectedMetric = METRICS[0];
  let selectedSquare = null;
  let candidates = [];
  let highlightedId = null;
  let generation = 0;
  const charts = new Map();

  // ---------------------------------------------------------------- Layout

  const heading = document.createElement("h3");
  heading.textContent = "Browse the whole recording";
  const intro = document.createElement("p");
  intro.textContent = `The plots compare the two models over all ${overview.n_events.toLocaleString("en-US")} spikes in the session. Each square counts spikes with similar values under both models; darker squares hold more spikes (log scale). Choose a square to open a recording window around one of its spikes below.`;
  const chartGrid = document.createElement("div");
  chartGrid.className = "comparison-charts";
  for (const metric of METRICS) {
    const chart = chartElement(metric, overview, pick);
    chartGrid.appendChild(chart.figure);
    charts.set(metric.name, chart.canvas);
  }
  const chartNote = document.createElement("p");
  chartNote.className = "note";
  chartNote.textContent =
    "The dashed diagonal marks equal values under both models, and dotted lines mark flag cutoffs. Squares far from the diagonal hold the spikes on which the models disagree. The chosen square is outlined, and the selected spike circled, in every plot.";

  const controls = document.createElement("div");
  controls.className = "comparison-controls";
  const labeled = (text, input) => {
    const label = document.createElement("label");
    label.append(text, input);
    return label;
  };
  const metricSelect = document.createElement("select");
  for (const metric of METRICS) {
    const option = document.createElement("option");
    option.value = metric.name;
    option.textContent = AXIS_LABEL[metric.name];
    metricSelect.appendChild(option);
  }
  const squareInput = () => {
    const input = document.createElement("input");
    input.type = "range";
    input.min = "1";
    input.max = String(grid);
    input.value = "1";
    return input;
  };
  const xInput = squareInput();
  const yInput = squareInput();
  const squareStatus = document.createElement("p");
  squareStatus.className = "comparison-status";
  squareStatus.setAttribute("role", "status");
  const spikeRange = document.createElement("input");
  spikeRange.type = "range";
  const spikeNumber = document.createElement("input");
  spikeNumber.type = "number";
  for (const input of [spikeRange, spikeNumber]) {
    input.min = "1";
    input.max = "1";
    input.value = "1";
    input.disabled = true;
  }
  const spikeStatus = document.createElement("p");
  spikeStatus.className = "comparison-status";
  spikeStatus.setAttribute("aria-live", "polite");
  const viewLink = document.createElement("a");
  viewLink.href = "#rec-view";
  viewLink.hidden = true;
  viewLink.textContent = "View the selected spike's recording ↓";
  const permalink = document.createElement("a");
  permalink.href = window.location.href;
  permalink.hidden = true;
  permalink.textContent = "Link to the selected spike";
  controls.append(
    labeled("Diagnostic ", metricSelect),
    labeled("Continuous value square ", xInput),
    labeled("Continuous–Fragmented value square ", yInput),
    squareStatus,
    labeled("Spike in this square ", spikeRange),
    labeled("Spike number in this square ", spikeNumber),
    spikeStatus,
    viewLink,
    permalink,
  );
  explorer.replaceChildren(heading, intro, chartGrid, chartNote, agreementTable(overview), controls);

  // ------------------------------------------------------------- Drawing

  /** The selected spike's plotted values under both models, if its window is shown. */
  function highlightValues(metricName) {
    if (!shown) return null;
    const models = Object.values(shown.payload.models);
    const local = models[0].events.id.indexOf(highlightedId);
    if (local < 0) return null;
    return models.map((model) => plotted(metricName, model.events[metricName][local]));
  }

  function draw() {
    for (const metric of METRICS) {
      const square = metric.name === selectedMetric.name ? selectedSquare : null;
      drawChart(charts.get(metric.name), overview, metric, square, highlightValues(metric.name));
    }
  }
  draw();

  // --------------------------------------------------------------- Data

  function loadIndex() {
    indexLoad ??= loadContainer("data/explorer/index.bin.gz")
      .then(({ meta, arrays }) => {
        if (
          meta.decode_cache_fingerprint !== overview.decode_cache_fingerprint ||
          meta.diagnostics_fingerprint !== overview.diagnostics_fingerprint ||
          meta.n_events !== overview.n_events
        ) {
          throw new Error("the spike index does not match the session overview");
        }
        index = arrays;
        return arrays;
      })
      .catch((error) => {
        indexLoad = null;
        throw error;
      });
    return indexLoad;
  }

  function loadBlock(number) {
    if (!blockCache.has(number)) {
      const load = loadContainer(`data/explorer/blocks/${String(number).padStart(4, "0")}.bin.gz`)
        .then((block) => {
          if (block.meta.decode_cache_fingerprint !== overview.decode_cache_fingerprint) {
            throw new Error("a recording block does not match the session overview");
          }
          return block;
        });
      load.catch(() => blockCache.delete(number));
      blockCache.set(number, load);
      if (blockCache.size > BLOCK_CACHE_SIZE) blockCache.delete(blockCache.keys().next().value);
    }
    return blockCache.get(number);
  }

  /** The window (first of its two blocks) that keeps the spike near its middle. */
  function windowFor(timeBin) {
    const block = Math.floor(timeBin / B);
    const start = timeBin - block * B < B / 2 ? block - 1 : block;
    return Math.min(overview.n_blocks - overview.window_blocks, Math.max(0, start));
  }

  // ---------------------------------------------------------- Selection

  function setLink(id) {
    const url = new URL(window.location.href);
    url.searchParams.set("recording_event", String(id));
    window.history.replaceState(null, "", url);
    permalink.href = url.href;
    permalink.hidden = false;
  }

  function describeSquare() {
    const { max, counts } = overview.metrics[selectedMetric.name];
    const precision = max <= 1 ? 3 : 1;
    const interval = (square) =>
      `${((square * max) / grid).toFixed(precision)}–${(((square + 1) * max) / grid).toFixed(precision)}`;
    const count = counts[selectedSquare];
    squareStatus.textContent = `${count.toLocaleString("en-US")} spike${count === 1 ? "" : "s"} in this square: ${AXIS_LABEL[selectedMetric.name]} ${interval(Math.floor(selectedSquare / grid))} under the Continuous model and ${interval(selectedSquare % grid)} under the Continuous–Fragmented model.`;
  }

  function describeAxes() {
    const { max } = overview.metrics[selectedMetric.name];
    const precision = max <= 1 ? 3 : 1;
    for (const input of [xInput, yInput]) {
      const square = Number(input.value) - 1;
      input.setAttribute(
        "aria-valuetext",
        `${((square * max) / grid).toFixed(precision)} to ${(((square + 1) * max) / grid).toFixed(precision)}`,
      );
    }
  }

  function describeSpike(position) {
    const id = candidates[position];
    const seconds = index.time_bin[id] * overview.time_step;
    const text = `Spike ${position + 1} of ${candidates.length.toLocaleString("en-US")} in this square: unit ${index.cell[id] + 1}, ${seconds.toFixed(3)} s into the session.`;
    spikeStatus.textContent = text;
    spikeRange.setAttribute("aria-valuetext", text);
  }

  /** Choose `square` of the selected diagnostic; its spikes come from the index. */
  function setSquare(square) {
    selectedSquare = square;
    xInput.value = String(Math.floor(square / grid) + 1);
    yInput.value = String((square % grid) + 1);
    describeAxes();
    describeSquare();
    const squares = index[`square_${selectedMetric.name}`];
    candidates = [];
    for (let id = 0; id < squares.length; id += 1) if (squares[id] === square) candidates.push(id);
    if (candidates.length !== overview.metrics[selectedMetric.name].counts[square]) {
      throw new Error("the square's count does not match the spike index");
    }
    for (const input of [spikeRange, spikeNumber]) {
      input.max = String(Math.max(1, candidates.length));
      input.disabled = candidates.length === 0;
    }
  }

  function setPosition(position) {
    spikeRange.value = String(position + 1);
    spikeNumber.value = String(position + 1);
    describeSpike(position);
  }

  /** Open spike `id` in the player, loading its window if another is shown. */
  async function showSpike(id) {
    const request = ++generation;
    highlightedId = id;
    setLink(id);
    draw();
    const windowIndex = windowFor(index.time_bin[id]);
    if (shown?.window === windowIndex && player?.selectEventId(id)) return;
    spikeStatus.textContent += " Loading its recording window…";
    try {
      const blocks = await Promise.all(
        Array.from({ length: overview.window_blocks }, (_, i) => loadBlock(windowIndex + i)),
      );
      await paperLoad;
      if (request !== generation) return;
      const payload = windowPayload(blocks, windowIndex, overview, shared);
      player?.destroy();
      shown = { window: windowIndex, payload };
      player = renderRecording(root, payload, manifest, {
        initialEventId: id,
        onEventSelect: followPlayer,
      });
      viewLink.hidden = false;
      draw();
    } catch (error) {
      if (request === generation) {
        spikeStatus.textContent = `Could not load this recording window (${error.message}).`;
        console.error(error);
      }
    }
  }

  /** Keep the plots, controls, and link on the spike the player settles on. */
  function followPlayer(id) {
    if (!index) return;
    highlightedId = id;
    setLink(id);
    const square = index[`square_${selectedMetric.name}`][id];
    if (square !== selectedSquare) setSquare(square);
    setPosition(candidates.indexOf(id));
    draw();
  }

  async function chooseSquare(square, requestedId = null) {
    const request = ++generation;
    try {
      await loadIndex();
    } catch (error) {
      squareStatus.textContent = `Could not load the spike index (${error.message}).`;
      console.error(error);
      return;
    }
    if (request !== generation) return;
    setSquare(square);
    draw();
    if (!candidates.length) {
      spikeStatus.textContent = "No spikes in this square; choose a colored one.";
      return;
    }
    const position =
      requestedId === null ? Math.floor(candidates.length / 2) : candidates.indexOf(requestedId);
    setPosition(position);
    await showSpike(candidates[position]);
  }

  function pick(metricName, xFraction, yFraction) {
    selectedMetric = METRICS.find((metric) => metric.name === metricName);
    metricSelect.value = metricName;
    const clamp = (value) => Math.min(grid - 1, Math.max(0, Math.floor(value * grid)));
    const [x, y] = [clamp(xFraction), clamp(yFraction)];
    // A click on an empty square opens the nearest square with spikes.
    const { counts } = overview.metrics[metricName];
    let square = x * grid + y;
    if (!counts[square]) {
      let best = Infinity;
      counts.forEach((count, candidate) => {
        const distance = (Math.floor(candidate / grid) - x) ** 2 + ((candidate % grid) - y) ** 2;
        if (count && distance < best) [best, square] = [distance, candidate];
      });
    }
    chooseSquare(square);
  }

  const squareFromInputs = () => (Number(xInput.value) - 1) * grid + Number(yInput.value) - 1;
  const settleSquare = debounce(() => chooseSquare(squareFromInputs()), SETTLE_MS);
  const settleSpike = debounce(() => showSpike(candidates[Number(spikeRange.value) - 1]), SETTLE_MS);
  metricSelect.addEventListener("change", () => {
    selectedMetric = METRICS.find((metric) => metric.name === metricSelect.value);
    describeAxes();
    chooseSquare(squareFromInputs());
  });
  for (const input of [xInput, yInput]) {
    input.addEventListener("input", describeAxes);
    input.addEventListener("change", settleSquare);
  }
  spikeRange.addEventListener("input", () => setPosition(Number(spikeRange.value) - 1));
  spikeRange.addEventListener("change", settleSpike);
  spikeNumber.addEventListener("change", () => {
    const position = Math.min(candidates.length, Math.max(1, Math.round(Number(spikeNumber.value)) || 1)) - 1;
    setPosition(position);
    showSpike(candidates[position]);
  });
  describeAxes();

  // ?recording_event=<id> opens that spike in its square of the first diagnostic.
  const requested = new URLSearchParams(window.location.search).get("recording_event");
  const id = Number(requested);
  if (requested !== null && Number.isInteger(id) && id >= 0 && id < overview.n_events) {
    loadIndex()
      .then(() => chooseSquare(index[`square_${selectedMetric.name}`][id], id))
      .catch((error) => {
        squareStatus.textContent = `Could not open the linked spike (${error.message}).`;
      });
  } else {
    squareStatus.textContent =
      "Choose a colored square, or use the diagnostic and square controls, to browse the session.";
  }
}
