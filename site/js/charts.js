// Drawing primitives: time-aligned canvas tracks with a shared cursor, and an
// SVG chart of a prediction and a spike likelihood over position.

import { argmax } from "./data.js";

const SVG_NS = "http://www.w3.org/2000/svg";

// A tap moves the cursor only if the pointer travelled less than this (px);
// longer touch movements are scrolls.
const TAP_SLOP = 10;

// The stylesheet's tokens are fixed for the page's lifetime, so read each once.
const cssVars = new Map();

export function cssVar(name) {
  let value = cssVars.get(name);
  if (value === undefined) {
    value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    cssVars.set(name, value);
  }
  return value;
}

function svg(tag, attributes = {}, parent = null) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (parent) parent.appendChild(element);
  return element;
}

function niceTicks(min, max, target = 5) {
  const span = max - min;
  if (!(span > 0)) return [min];
  const raw = span / target;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => span / s <= target);
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step) {
    ticks.push(Number(t.toFixed(10)));
  }
  return ticks;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/** Call `onChange` whenever the device pixel ratio changes; returns a remover. */
function watchPixelRatio(onChange) {
  let query = null;
  const handle = () => {
    onChange();
    listen();
  };
  const listen = () => {
    query = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
    query.addEventListener("change", handle, { once: true });
  };
  listen();
  return () => query?.removeEventListener("change", handle);
}

// ---------------------------------------------------------------------------
// Heatmap bitmaps
// ---------------------------------------------------------------------------

function hexToRgb(hex) {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

/**
 * Render rows of uint8 values (one row per time step, bins along the row) to
 * an offscreen canvas: time on x, position on y (increasing upward).
 *
 * Without `scale`, each row is colored relative to its own maximum. With
 * `scale = {rowMax, range}` (which a decodeHeatmap result carries itself),
 * values are recovered as row / 255 * rowMax[t] and colored on the shared
 * `range`, as in the paper's heatmaps.
 */
export function heatmapBitmap(rows, lut, scale = rows.range ? rows : null) {
  const { nRows, nBins } = rows;
  const canvas = document.createElement("canvas");
  canvas.width = nRows;
  canvas.height = nBins;
  const context = canvas.getContext("2d");
  const image = context.createImageData(nRows, nBins);
  const colors = lut.map(hexToRgb);
  const [low, high] = scale ? scale.range : [0, 1];
  const span = high > low ? high - low : 1;
  for (let t = 0; t < nRows; t += 1) {
    const row = rows.row(t);
    const factor = scale ? scale.rowMax[t] / 255 : 1 / 255;
    for (let b = 0; b < nBins; b += 1) {
      const level = Math.min(1, Math.max(0, (row[b] * factor - low) / span));
      const color = colors[Math.round(level * (colors.length - 1))];
      const offset = ((nBins - 1 - b) * nRows + t) * 4;
      image.data[offset] = color[0];
      image.data[offset + 1] = color[1];
      image.data[offset + 2] = color[2];
      image.data[offset + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  return canvas;
}

// ---------------------------------------------------------------------------
// Track stack
// ---------------------------------------------------------------------------

/**
 * A vertical stack of canvas tracks sharing one time axis and a cursor. For
 * assistive technology the stack is a slider over the time range whose value
 * the owner sets with `setValue(time, text)`. Owners set it only for discrete
 * selections: a focused slider's screen reader speaks every change.
 *
 * tracks: [{ label, note?, top, bottom, height, draw(ctx, width, height, xOf) }]
 * range:  [t0, t1] in seconds (or the unit tickLabel names).
 * onCursor(time, pressed): mouse hover or drag (unless hover is false; pressed
 *   false), or a press, a release, or a tap (not a scroll) on touch screens
 *   (pressed true).
 * onKey(key): ArrowLeft/ArrowRight/Home/End while the stack has focus;
 *   ArrowDown and ArrowUp arrive as ArrowLeft and ArrowRight, as on a slider.
 * tickLabel(time): axis tick text; seconds by default.
 */
export class TrackStack {
  constructor(
    container,
    { tracks, range, onCursor, onKey, ariaLabel, hover = true, tickLabel = null },
  ) {
    this.tracks = tracks;
    this.range = range;
    this.onCursor = onCursor;
    this.tickLabel = tickLabel ?? ((t) => `${Number(t.toFixed(3))} s`);
    this.root = document.createElement("div");
    this.root.className = "stack";
    this.root.tabIndex = 0;
    this.root.setAttribute("role", "slider");
    this.root.setAttribute("aria-label", ariaLabel);
    this.root.setAttribute("aria-valuemin", String(range[0]));
    this.root.setAttribute("aria-valuemax", String(range[1]));
    this.root.setAttribute("aria-valuenow", String(range[0]));
    this.canvases = [];

    for (const track of [...tracks, { axis: true, height: 22 }]) {
      const row = document.createElement("div");
      row.className = "track";
      const label = document.createElement("div");
      label.className = "track-label";
      if (!track.axis) {
        const note = track.note ? `<span class="direction">${track.note}</span>` : "";
        label.innerHTML = `<span>${track.top ?? ""}</span><span><span class="name">${track.label}</span>${note}</span><span>${track.bottom ?? ""}</span>`;
      }
      const canvas = document.createElement("canvas");
      canvas.style.height = `${track.height}px`;
      row.append(label, canvas);
      this.root.appendChild(row);
      this.canvases.push({ canvas, track });
    }
    this.cursor = document.createElement("div");
    this.cursor.className = "cursor";
    this.root.appendChild(this.cursor);
    container.appendChild(this.root);

    const timeAt = (clientX) => {
      const rect = this.canvases[0].canvas.getBoundingClientRect();
      const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      return this.range[0] + fraction * (this.range[1] - this.range[0]);
    };
    let touchStart = null;
    this.root.addEventListener("pointerdown", (event) => {
      if (event.pointerType === "touch") {
        touchStart = { x: event.clientX, y: event.clientY };
        return;
      }
      // Capture, so a drag released outside the tracks still ends here.
      this.root.setPointerCapture(event.pointerId);
      this.onCursor(timeAt(event.clientX), true);
    });
    this.root.addEventListener("pointermove", (event) => {
      if (hover && event.pointerType !== "touch") this.onCursor(timeAt(event.clientX), false);
    });
    this.root.addEventListener("pointerup", (event) => {
      // A release ends a drag: the selection it leaves is a discrete one.
      if (event.pointerType !== "touch") {
        this.onCursor(timeAt(event.clientX), true);
        return;
      }
      if (!touchStart) return;
      const moved = Math.hypot(event.clientX - touchStart.x, event.clientY - touchStart.y);
      touchStart = null;
      if (moved < TAP_SLOP) this.onCursor(timeAt(event.clientX), true);
    });
    this.root.addEventListener("pointercancel", () => {
      touchStart = null;
    });
    const keys = {
      ArrowRight: "ArrowRight",
      ArrowUp: "ArrowRight",
      ArrowLeft: "ArrowLeft",
      ArrowDown: "ArrowLeft",
      Home: "Home",
      End: "End",
    };
    this.root.addEventListener("keydown", (event) => {
      if (event.key in keys) {
        event.preventDefault();
        onKey(keys[event.key]);
      }
    });

    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(this.root);
    this.stopWatchingRatio = watchPixelRatio(() => this.draw());
  }

  destroy() {
    this.resizeObserver.disconnect();
    this.stopWatchingRatio();
  }

  xOf(width) {
    const [t0, t1] = this.range;
    return (t) => ((t - t0) / (t1 - t0)) * width;
  }

  draw() {
    for (const { canvas, track } of this.canvases) {
      const width = canvas.clientWidth;
      const height = track.height;
      if (width === 0) continue;
      const ratio = window.devicePixelRatio || 1;
      // Assigning a canvas size reallocates its bitmap, so only do it on change.
      const pixelWidth = Math.round(width * ratio);
      const pixelHeight = Math.round(height * ratio);
      if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
      if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
      const context = canvas.getContext("2d");
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);
      if (track.axis) this.drawAxis(context, width);
      else track.draw(context, width, height, this.xOf(width));
    }
    this.placeCursor(this.cursorTime);
  }

  drawAxis(context, width) {
    const xOf = this.xOf(width);
    context.strokeStyle = cssVar("--border");
    context.fillStyle = cssVar("--text-muted");
    context.font = `11px ${cssVar("--font")}`;
    context.textAlign = "center";
    context.textBaseline = "top";
    context.beginPath();
    context.moveTo(0, 0.5);
    context.lineTo(width, 0.5);
    for (const tick of niceTicks(this.range[0], this.range[1], Math.max(3, width / 90))) {
      const x = Math.round(xOf(tick)) + 0.5;
      context.moveTo(x, 0);
      context.lineTo(x, 4);
      const text = this.tickLabel(tick);
      context.fillText(text, Math.min(width - 14, Math.max(14, x)), 6);
    }
    context.stroke();
  }

  /** The slider's value, a time in the range, and its spoken text, e.g. the selected spike. */
  setValue(time, text) {
    this.root.setAttribute("aria-valuenow", String(Number(time.toFixed(3))));
    this.root.setAttribute("aria-valuetext", text);
  }

  placeCursor(time) {
    this.cursorTime = time;
    if (time === undefined || time === null) {
      this.cursor.style.display = "none";
      return;
    }
    const first = this.canvases[0].canvas;
    const left = first.offsetLeft + this.xOf(first.clientWidth)(time);
    this.cursor.style.left = `${left}px`;
    this.cursor.style.display = "block";
  }
}

// ---------------------------------------------------------------------------
// Track painters
// ---------------------------------------------------------------------------

/** Position expressed in bin-index units, interpolating between bin centers. */
function fractionalIndex(bins, value) {
  const last = bins.length - 1;
  if (value <= bins[0]) return 0;
  if (value >= bins[last]) return last;
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (bins[mid] <= value) lo = mid;
    else hi = mid;
  }
  return lo + (value - bins[lo]) / (bins[hi] - bins[lo]);
}

/** Map a position to a heatmap's y coordinate (bins drawn at equal heights). */
export function positionScale(bins, height) {
  return (position) => height - ((fractionalIndex(bins, position) + 0.5) / bins.length) * height;
}

export function paintHeatmap(context, bitmap, width, height) {
  context.imageSmoothingEnabled = true;
  context.drawImage(bitmap, 0, 0, width, height);
}

// Narrowest painted spike column, in pixels.
const MIN_COLUMN_WIDTH = 1.5;

/**
 * Paint only the time steps where `mask(t)` holds, each at least
 * MIN_COLUMN_WIDTH pixels wide, over a black background. Isolated spikes stay
 * visible even when a time step is narrower than a pixel.
 */
export function paintColumns(context, bitmap, times, mask, xOf, width, height) {
  context.fillStyle = "#000000";
  context.fillRect(0, 0, width, height);
  context.imageSmoothingEnabled = false;
  const step = times.length > 1 ? times[1] - times[0] : 0;
  for (let t = 0; t < times.length; t += 1) {
    if (!mask(t)) continue;
    const x = xOf(times[t]);
    const w = Math.max(MIN_COLUMN_WIDTH, xOf(times[t] + step) - x);
    context.drawImage(bitmap, t, 0, 1, bitmap.height, x, 0, w, height);
  }
}

/** Line through (t, position) pairs; gaps where position is null. */
export function paintPositionLine(context, times, positions, xOf, yOf, color, lineWidth = 1.6) {
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.lineJoin = "round";
  context.beginPath();
  let drawing = false;
  for (let i = 0; i < times.length; i += 1) {
    if (positions[i] === null) {
      drawing = false;
      continue;
    }
    const x = xOf(times[i]);
    const y = yOf(positions[i]);
    if (drawing) context.lineTo(x, y);
    else context.moveTo(x, y);
    drawing = true;
  }
  context.stroke();
}

export function paintShading(context, spans, xOf, height, color) {
  context.fillStyle = color;
  for (const [a, b] of spans) context.fillRect(xOf(a), 0, xOf(b) - xOf(a), height);
}

export function paintThreshold(context, y, width) {
  context.strokeStyle = cssVar("--threshold");
  context.lineWidth = 1;
  context.setLineDash([4, 3]);
  context.beginPath();
  context.moveTo(0, Math.round(y) + 0.5);
  context.lineTo(width, Math.round(y) + 0.5);
  context.stroke();
  context.setLineDash([]);
}

/** Dots at (t, value); style(i) returns {fill: bool, alpha}. */
export function paintDots(context, times, values, xOf, yOf, color, style) {
  for (let i = 0; i < times.length; i += 1) {
    const { fill, alpha = 1, radius = 2.6 } = style(i);
    const x = xOf(times[i]);
    const y = yOf(values[i]);
    context.globalAlpha = alpha;
    context.beginPath();
    context.arc(x, y, radius, 0, 2 * Math.PI);
    if (fill) {
      context.fillStyle = color;
      context.fill();
    } else {
      context.strokeStyle = color;
      context.lineWidth = 1.2;
      context.stroke();
    }
  }
  context.globalAlpha = 1;
}

// ---------------------------------------------------------------------------
// Distribution chart (SVG)
// ---------------------------------------------------------------------------

// Distinguishes the radio groups of several cell strips on one page.
let cellPickerCount = 0;

/**
 * Distributions over position (e.g., a prediction and a spike likelihood),
 * each scaled to its own maximum unless a shared `scaleMax` is given, with
 * optional HPD bands, a position marker, and a strip of place fields for
 * choosing the firing cell.
 *
 * The SVG is laid out at its container's pixel width, so text keeps its CSS
 * size on narrow screens. Curves and bands break across gaps in the position
 * grid (e.g., between linearized track segments).
 *
 * The SVG is an image whose text alternative, `title` followed by the peak of
 * each named series, the extent of each named band, and the marker, is
 * rewritten on every update; positions are given in `unit`.
 *
 * axis: false omits the position axis (for charts stacked above another).
 *
 * cellStrip: { label, state() -> {rates, selectable, centers, selected},
 *              onSelect(cell) }. The pointer selects a field in the SVG; the
 *              keyboard and screen readers use a visually hidden native radio
 *              group after it, whose focus the SVG outlines.
 */
export class DistributionChart {
  constructor(
    container,
    { positionBins, xLabel, title, unit, plotHeight = 150, cellStrip = null, axis = true },
  ) {
    this.container = container;
    this.bins = positionBins;
    this.xLabel = xLabel;
    this.title = title;
    this.unit = unit;
    this.plotHeight = plotHeight;
    this.cellStrip = cellStrip;
    this.showAxis = axis;
    this.margin = { left: 12, right: 12, top: 10 };
    // Below the plot: two HPD bands, the axis, tick labels, and the axis title.
    this.axisBlock = axis ? 60 : 4;
    this.stripHeight = 34;
    this.xMin = positionBins[0];
    this.xMax = positionBins[positionBins.length - 1];
    const spacing = median(positionBins.slice(1).map((b, i) => b - positionBins[i]));
    this.binWidth = spacing;
    this.segmentStart = positionBins.map(
      (b, i) => i === 0 || b - positionBins[i - 1] > 1.5 * spacing,
    );

    this.root = svg("svg", { class: "dist-chart", role: "img", "aria-label": title });
    container.appendChild(this.root);
    this.layers = {
      bands: svg("g", {}, this.root),
      areas: svg("g", {}, this.root),
      marker: svg("g", {}, this.root),
      axis: svg("g", { class: "axis" }, this.root),
      cells: svg("g", {}, this.root),
    };
    this.width = 0;
    this.last = null;
    this.cellKey = null;
    this.pickerKey = null;
    this.focusedCell = null;
    this.resize();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
  }

  destroy() {
    this.resizeObserver.disconnect();
  }

  resize() {
    const width = Math.max(260, Math.round(this.container.clientWidth || 640));
    if (width === this.width) return;
    this.width = width;
    const height =
      this.margin.top +
      this.plotHeight +
      this.axisBlock +
      (this.cellStrip ? this.stripHeight + 30 : 0);
    this.root.setAttribute("viewBox", `0 0 ${width} ${height}`);
    this.root.setAttribute("height", height);
    this.drawAxis();
    this.cellKey = null;
    if (this.last) this.update(this.last);
  }

  x(value) {
    const { left, right } = this.margin;
    return left + ((value - this.xMin) / (this.xMax - this.xMin)) * (this.width - left - right);
  }

  drawAxis() {
    this.layers.axis.replaceChildren();
    const y = this.margin.top + this.plotHeight + 22;
    if (!this.showAxis) {
      const base = this.margin.top + this.plotHeight;
      svg("line", { x1: this.x(this.xMin), x2: this.x(this.xMax), y1: base, y2: base }, this.layers.axis);
      return;
    }
    svg("line", { x1: this.x(this.xMin), x2: this.x(this.xMax), y1: y, y2: y }, this.layers.axis);
    const target = Math.max(3, Math.min(6, Math.floor(this.width / 80)));
    for (const tick of niceTicks(this.xMin, this.xMax, target)) {
      svg("line", { x1: this.x(tick), x2: this.x(tick), y1: y, y2: y + 4 }, this.layers.axis);
      const label = svg("text", { x: this.x(tick), y: y + 16, "text-anchor": "middle" });
      label.textContent = String(tick);
      this.layers.axis.appendChild(label);
    }
    const title = svg("text", { x: this.width / 2, y: y + 32, "text-anchor": "middle" });
    title.textContent = this.xLabel;
    this.layers.axis.appendChild(title);
  }

  /** Path data for `values`: `open` traces the curve, `closed` also fills to the baseline. */
  curvePaths(values, scaleMax = null) {
    const max = scaleMax ?? Math.max(...values);
    const base = this.margin.top + this.plotHeight;
    const yOf = (v) => base - (max > 0 ? v / max : 0) * (this.plotHeight - 4);
    let open = "";
    let closed = "";
    let segmentFirst = 0;
    const closeSegment = (last) => {
      const right = this.x(this.bins[last]).toFixed(2);
      const left = this.x(this.bins[segmentFirst]).toFixed(2);
      closed += `L${right},${base}L${left},${base}Z`;
    };
    values.forEach((v, i) => {
      let point = "L";
      if (this.segmentStart[i]) {
        if (i > 0) closeSegment(i - 1);
        segmentFirst = i;
        point = "M";
      }
      point += `${this.x(this.bins[i]).toFixed(2)},${yOf(v).toFixed(2)}`;
      open += point;
      closed += point;
    });
    closeSegment(values.length - 1);
    return { open, closed };
  }

  /** [first, last] bin indices of each contiguous run of `mask`, split at grid gaps. */
  runs(mask) {
    const runs = [];
    let start = null;
    for (let i = 0; i < mask.length; i += 1) {
      if (start !== null && (!mask[i] || this.segmentStart[i])) {
        runs.push([start, i - 1]);
        start = null;
      }
      if (mask[i] && start === null) start = i;
    }
    if (start !== null) runs.push([start, mask.length - 1]);
    return runs;
  }

  /** One rect per contiguous run of in-region bins, so bands have no seams. */
  drawBand(mask, color, y) {
    const half = (this.x(this.bins[0] + this.binWidth) - this.x(this.bins[0])) / 2;
    for (const [first, last] of this.runs(mask)) {
      const x0 = this.x(this.bins[first]) - half;
      const x1 = this.x(this.bins[last]) + half;
      svg("rect", { x: x0, y, width: x1 - x0, height: 5, fill: color }, this.layers.bands);
    }
  }

  /** A position as text, to at most one decimal. */
  place(value) {
    return String(Number(value.toFixed(1)));
  }

  /** The chart's text alternative for one update's state. */
  describe({ series, bands, marker }) {
    const parts = [];
    for (const { name, values } of series) {
      if (!name) continue;
      const peak = argmax(values);
      parts.push(
        values.every((v) => v === values[peak])
          ? `${name} is flat`
          : `${name} peaks at ${this.place(this.bins[peak])} ${this.unit}`,
      );
    }
    for (const { name, mask } of bands) {
      if (!name) continue;
      const spans = this.runs(mask).map(([first, last]) =>
        first === last
          ? this.place(this.bins[first])
          : `${this.place(this.bins[first])}–${this.place(this.bins[last])}`,
      );
      parts.push(`${name} ${spans.length ? `spans ${spans.join(", ")} ${this.unit}` : "is empty"}`);
    }
    if (marker !== null) parts.push(`animal at ${this.place(marker)} ${this.unit}`);
    if (!parts.length) return this.title;
    // Units such as "a.u." already end the sentence.
    const text = `${this.title}: ${parts.join("; ")}`;
    return text.endsWith(".") ? text : `${text}.`;
  }

  /**
   * series: [{values, color, name?, dashed?, filled? (default true), width? (default 2)}],
   * bands: [{mask, color, name?}], marker: the animal's position | null,
   * scaleMax: value drawn at full height for every series | null.
   * Only named series and bands enter the text alternative.
   */
  update(state) {
    this.last = state;
    const { series, bands = [], marker = null, scaleMax = null } = state;
    this.root.setAttribute("aria-label", this.describe({ series, bands, marker }));
    for (const layer of ["bands", "areas", "marker"]) this.layers[layer].replaceChildren();
    for (const { values, color, dashed = false, filled = true, width = 2 } of series) {
      const { open, closed } = this.curvePaths(values, scaleMax);
      if (filled) svg("path", { d: closed, fill: color, "fill-opacity": 0.12 }, this.layers.areas);
      svg(
        "path",
        {
          d: open,
          fill: "none",
          stroke: color,
          "stroke-width": width,
          "stroke-linejoin": "round",
          ...(dashed ? { "stroke-dasharray": "5 4" } : {}),
        },
        this.layers.areas,
      );
    }
    bands.forEach(({ mask, color }, index) => {
      this.drawBand(mask, color, this.margin.top + this.plotHeight + 4 + index * 7);
    });
    if (marker !== null) {
      svg(
        "line",
        {
          x1: this.x(marker),
          x2: this.x(marker),
          y1: this.margin.top,
          y2: this.margin.top + this.plotHeight,
          stroke: cssVar("--position"),
          "stroke-width": 2,
        },
        this.layers.marker,
      );
    }
    if (this.cellStrip) this.drawCells();
  }

  /**
   * Build the radio group when the cell set changes and the strip when the
   * cell set or width changes; otherwise restyle both.
   */
  drawCells() {
    const { rates, selectable, centers, selected } = this.cellStrip.state();
    const order = [...selectable].sort((a, b) => centers[a] - centers[b]);
    const pickerKey = order.join(",");
    if (pickerKey !== this.pickerKey) {
      this.pickerKey = pickerKey;
      this.buildPicker(order, centers);
    }
    const key = `${pickerKey}@${this.width}`;
    if (key !== this.cellKey) {
      this.cellKey = key;
      this.buildCells(rates, order, centers);
    }
    for (const { cell, rect, path } of this.cells) {
      const isSelected = cell === selected;
      rect.classList.toggle("focused", cell === this.focusedCell);
      path.setAttribute("stroke", isSelected ? cssVar("--likelihood") : cssVar("--curve-muted"));
      path.setAttribute("stroke-width", isSelected ? 2.5 : 1.2);
    }
    for (const [cell, input] of this.radios) input.checked = cell === selected;
  }

  cellLabel(cell, centers) {
    return `Cell ${cell + 1}, field centered at ${centers[cell]} ${this.unit}`;
  }

  /** The native radio group, in field order, after the SVG. */
  buildPicker(order, centers) {
    this.picker?.remove();
    this.focusedCell = null;
    const name = `cell-picker-${(cellPickerCount += 1)}`;
    const fieldset = document.createElement("fieldset");
    fieldset.className = "sr-only";
    const legend = document.createElement("legend");
    legend.textContent = this.cellStrip.label;
    fieldset.appendChild(legend);
    this.radios = new Map();
    for (const cell of order) {
      const input = document.createElement("input");
      input.type = "radio";
      input.name = name;
      input.value = String(cell);
      input.addEventListener("change", () => this.cellStrip.onSelect(cell));
      input.addEventListener("focus", () => {
        this.focusedCell = cell;
        this.drawCells();
      });
      input.addEventListener("blur", () => {
        this.focusedCell = null;
        this.drawCells();
      });
      const label = document.createElement("label");
      label.append(input, ` ${this.cellLabel(cell, centers)}`);
      fieldset.appendChild(label);
      this.radios.set(cell, input);
    }
    this.container.appendChild(fieldset);
    this.picker = fieldset;
  }

  /** The strip of place fields, `order` sorted by field center, with pointer hit areas. */
  buildCells(rates, order, centers) {
    this.layers.cells.replaceChildren();
    const top = this.margin.top + this.plotHeight + this.axisBlock + 22;
    const caption = svg("text", { x: this.margin.left, y: top - 8 }, this.layers.cells);
    caption.textContent = `${this.cellStrip.label}: click a field, or use the arrow keys`;
    // Curves first, then the hit areas on top of them.
    const curves = svg("g", { "pointer-events": "none" }, this.layers.cells);
    const hits = svg("g", {}, this.layers.cells);
    // Each hit area runs between the midpoints to the neighboring field
    // centers, so closely spaced fields stay individually selectable.
    const xs = order.map((cell) => this.x(centers[cell]));
    const bounds = xs.map((x, i) => {
      const gapLeft = i > 0 ? x - xs[i - 1] : null;
      const gapRight = i < xs.length - 1 ? xs[i + 1] - x : null;
      const outer = Math.min(24, (gapLeft ?? gapRight ?? 48) / 2);
      return [x - (gapLeft === null ? outer : gapLeft / 2), x + (gapRight === null ? outer : gapRight / 2)];
    });
    this.cells = order.map((cell, i) => {
      const column = rates.map((row) => row[cell]);
      const max = Math.max(...column);
      let d = "";
      column.forEach((v, b) => {
        const y = top + this.stripHeight - (v / max) * this.stripHeight;
        d += `${b === 0 ? "M" : "L"}${this.x(this.bins[b]).toFixed(1)},${y.toFixed(1)}`;
      });
      const path = svg("path", { d, fill: "none" }, curves);
      const [x0, x1] = bounds[i];
      const rect = svg(
        "rect",
        {
          class: "cell-hit",
          x: x0,
          y: top - 4,
          width: Math.max(1, x1 - x0),
          height: this.stripHeight + 8,
          fill: "transparent",
        },
        hits,
      );
      svg("title", {}, rect).textContent = this.cellLabel(cell, centers);
      rect.addEventListener("click", () => this.cellStrip.onSelect(cell));
      return { cell, rect, path };
    });
  }
}
