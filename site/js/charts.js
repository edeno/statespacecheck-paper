// Drawing primitives: time-aligned canvas tracks with a shared cursor, and an
// SVG chart of a prediction and a spike likelihood over position.

const SVG_NS = "http://www.w3.org/2000/svg";

// A tap moves the cursor only if the pointer travelled less than this (px);
// longer touch movements are scrolls.
const TAP_SLOP = 10;

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
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
 * `scale = {rowMax, range}`, values are recovered as row / 255 * rowMax[t]
 * and colored on the shared `range`, as in the paper's heatmaps.
 */
export function heatmapBitmap(rows, lut, scale = null) {
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
 * A vertical stack of canvas tracks sharing one time axis and a cursor.
 *
 * tracks: [{ label, note?, top, bottom, height, draw(ctx, width, height, xOf) }]
 * range:  [t0, t1] in seconds.
 * onCursor(time): mouse hover/press, or a tap (not a scroll) on touch screens.
 * onKey(key): ArrowLeft/ArrowRight/Home/End while the stack has focus.
 */
export class TrackStack {
  constructor(container, { tracks, range, onCursor, onKey, ariaLabel }) {
    this.tracks = tracks;
    this.range = range;
    this.onCursor = onCursor;
    this.root = document.createElement("div");
    this.root.className = "stack";
    this.root.tabIndex = 0;
    this.root.setAttribute("role", "group");
    this.root.setAttribute("aria-label", ariaLabel);
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
      if (event.pointerType === "touch") touchStart = { x: event.clientX, y: event.clientY };
      else this.onCursor(timeAt(event.clientX));
    });
    this.root.addEventListener("pointermove", (event) => {
      if (event.pointerType !== "touch") this.onCursor(timeAt(event.clientX));
    });
    this.root.addEventListener("pointerup", (event) => {
      if (event.pointerType !== "touch" || !touchStart) return;
      const moved = Math.hypot(event.clientX - touchStart.x, event.clientY - touchStart.y);
      touchStart = null;
      if (moved < TAP_SLOP) this.onCursor(timeAt(event.clientX));
    });
    this.root.addEventListener("pointercancel", () => {
      touchStart = null;
    });
    this.root.addEventListener("keydown", (event) => {
      if (["ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        onKey(event.key);
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
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
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
      const text = `${Number(tick.toFixed(3))} s`;
      context.fillText(text, Math.min(width - 14, Math.max(14, x)), 6);
    }
    context.stroke();
  }

  placeCursor(time) {
    this.cursorTime = time;
    if (time === undefined || time === null) {
      this.cursor.style.display = "none";
      return;
    }
    const first = this.canvases[0].canvas;
    const [t0, t1] = this.range;
    const left = first.offsetLeft + ((time - t0) / (t1 - t0)) * first.clientWidth;
    this.cursor.style.left = `${left}px`;
    this.cursor.style.display = "block";
  }
}

// ---------------------------------------------------------------------------
// Track painters
// ---------------------------------------------------------------------------

export function paintHeatmap(context, bitmap, width, height) {
  context.imageSmoothingEnabled = true;
  context.drawImage(bitmap, 0, 0, width, height);
}

/**
 * Paint only the time steps where `mask(t)` holds, each at least `minWidth`
 * pixels wide, over a black background. Isolated spikes stay visible even when
 * a time step is narrower than a pixel.
 */
export function paintColumns(context, bitmap, times, mask, xOf, width, height, minWidth = 1.5) {
  context.fillStyle = "#000000";
  context.fillRect(0, 0, width, height);
  context.imageSmoothingEnabled = false;
  const step = times.length > 1 ? times[1] - times[0] : 0;
  for (let t = 0; t < times.length; t += 1) {
    if (!mask(t)) continue;
    const x = xOf(times[t]);
    const w = Math.max(minWidth, xOf(times[t] + step) - x);
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

/**
 * Prediction and spike likelihood over position, each scaled to its own
 * maximum, with optional HPD bands, a position marker, and a keyboard- and
 * pointer-operable strip of place fields for choosing the firing cell.
 *
 * The SVG is laid out at its container's pixel width, so text keeps its CSS
 * size on narrow screens. Curves and bands break across gaps in the position
 * grid (e.g., between linearized track segments).
 *
 * cellStrip: { label, state() -> {rates, selectable, centers, selected},
 *              onSelect(cell) }
 */
export class DistributionChart {
  constructor(container, { positionBins, xLabel, plotHeight = 150, cellStrip = null }) {
    this.container = container;
    this.bins = positionBins;
    this.xLabel = xLabel;
    this.plotHeight = plotHeight;
    this.cellStrip = cellStrip;
    this.margin = { left: 12, right: 12, top: 10 };
    // Below the plot: two HPD bands, the axis, tick labels, and the axis title.
    this.axisBlock = 60;
    this.stripHeight = 34;
    this.xMin = positionBins[0];
    this.xMax = positionBins[positionBins.length - 1];
    const spacing = median(positionBins.slice(1).map((b, i) => b - positionBins[i]));
    this.binWidth = spacing;
    this.segmentStart = positionBins.map(
      (b, i) => i === 0 || b - positionBins[i - 1] > 1.5 * spacing,
    );

    this.root = svg("svg", { class: "dist-chart", role: "img" });
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

  curvePath(values, closed) {
    const max = Math.max(...values);
    const base = this.margin.top + this.plotHeight;
    const yOf = (v) => base - (max > 0 ? v / max : 0) * (this.plotHeight - 4);
    let d = "";
    let segmentFirst = 0;
    const closeSegment = (last) => {
      if (!closed) return;
      const right = this.x(this.bins[last]).toFixed(2);
      const left = this.x(this.bins[segmentFirst]).toFixed(2);
      d += `L${right},${base}L${left},${base}Z`;
    };
    values.forEach((v, i) => {
      if (this.segmentStart[i]) {
        if (i > 0) closeSegment(i - 1);
        segmentFirst = i;
        d += "M";
      } else {
        d += "L";
      }
      d += `${this.x(this.bins[i]).toFixed(2)},${yOf(v).toFixed(2)}`;
    });
    closeSegment(values.length - 1);
    return d;
  }

  /** One rect per contiguous run of in-region bins, so bands have no seams. */
  drawBand(mask, color, y) {
    const half = (this.x(this.bins[0] + this.binWidth) - this.x(this.bins[0])) / 2;
    let start = null;
    const close = (end) => {
      const x0 = this.x(this.bins[start]) - half;
      const x1 = this.x(this.bins[end]) + half;
      svg("rect", { x: x0, y, width: x1 - x0, height: 5, fill: color }, this.layers.bands);
      start = null;
    };
    for (let i = 0; i < mask.length; i += 1) {
      if (start !== null && (!mask[i] || this.segmentStart[i])) close(i - 1);
      if (mask[i] && start === null) start = i;
    }
    if (start !== null) close(mask.length - 1);
  }

  /** series: [{values, color}], bands: [{mask, color}], marker: position | null */
  update(state) {
    this.last = state;
    const { series, bands = [], marker = null } = state;
    for (const layer of ["bands", "areas", "marker"]) this.layers[layer].replaceChildren();
    for (const { values, color } of series) {
      svg(
        "path",
        { d: this.curvePath(values, true), fill: color, "fill-opacity": 0.12 },
        this.layers.areas,
      );
      svg(
        "path",
        {
          d: this.curvePath(values, false),
          fill: "none",
          stroke: color,
          "stroke-width": 2,
          "stroke-linejoin": "round",
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

  /** Build the cell radiogroup when the cell set or width changes; else restyle. */
  drawCells() {
    const { rates, selectable, centers, selected } = this.cellStrip.state();
    const key = `${selectable.join(",")}@${this.width}`;
    if (key !== this.cellKey) {
      this.cellKey = key;
      this.buildCells(rates, selectable, centers);
    }
    for (const { cell, rect, path } of this.cells) {
      const isSelected = cell === selected;
      rect.setAttribute("aria-checked", String(isSelected));
      rect.setAttribute("tabindex", isSelected ? "0" : "-1");
      path.setAttribute("stroke", isSelected ? cssVar("--likelihood") : cssVar("--curve-muted"));
      path.setAttribute("stroke-width", isSelected ? 2.5 : 1.2);
    }
  }

  buildCells(rates, selectable, centers) {
    this.layers.cells.replaceChildren();
    const top = this.margin.top + this.plotHeight + this.axisBlock + 22;
    const caption = svg("text", { x: this.margin.left, y: top - 8 }, this.layers.cells);
    caption.textContent = `${this.cellStrip.label}: click a field, or use the arrow keys`;
    // Curves first, then the hit areas on top of them.
    const curves = svg("g", { "pointer-events": "none" }, this.layers.cells);
    const group = svg(
      "g",
      { role: "radiogroup", "aria-label": this.cellStrip.label },
      this.layers.cells,
    );
    // Each hit area runs between the midpoints to the neighboring field
    // centers, so closely spaced fields stay individually selectable.
    const order = [...selectable].sort((a, b) => centers[a] - centers[b]);
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
      const label = `Cell ${cell + 1}, field centered at ${centers[cell]} a.u.`;
      const rect = svg(
        "rect",
        {
          class: "cell-hit",
          x: x0,
          y: top - 4,
          width: Math.max(1, x1 - x0),
          height: this.stripHeight + 8,
          fill: "transparent",
          role: "radio",
          "aria-label": label,
        },
        group,
      );
      svg("title", {}, rect).textContent = label;
      rect.addEventListener("click", () => this.cellStrip.onSelect(cell));
      rect.addEventListener("keydown", (event) => {
        const moves = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
        let next = null;
        if (event.key in moves) next = Math.min(order.length - 1, Math.max(0, i + moves[event.key]));
        else if (event.key === "Home") next = 0;
        else if (event.key === "End") next = order.length - 1;
        else if (event.key === "Enter" || event.key === " ") next = i;
        if (next === null) return;
        event.preventDefault();
        this.cellStrip.onSelect(order[next]);
        this.cells[next].rect.focus();
      });
      return { cell, rect, path };
    });
  }
}
