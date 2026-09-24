// Drawing primitives: time-aligned canvas tracks with a shared cursor, and an
// SVG chart of a prediction and a spike likelihood over position.

const SVG_NS = "http://www.w3.org/2000/svg";

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
  for (let t = 0; t < nRows; t += 1) {
    const row = rows.row(t);
    const factor = scale ? scale.rowMax[t] / 255 : 1 / 255;
    const [low, high] = scale ? scale.range : [0, 1];
    for (let b = 0; b < nBins; b += 1) {
      const level = Math.min(1, Math.max(0, (row[b] * factor - low) / (high - low)));
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
 * tracks: [{ label, top, bottom, height, draw(ctx, width, height, xOf) }]
 * range:  [t0, t1] in seconds.
 * onCursor(time) is called as the pointer moves; onStep(direction) on arrow keys.
 */
export class TrackStack {
  constructor(container, { tracks, range, onCursor, onStep, ariaLabel }) {
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
        label.innerHTML = `<span>${track.top ?? ""}</span><span class="name">${track.label}</span><span>${track.bottom ?? ""}</span>`;
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

    const plotArea = () => this.canvases[0].canvas.getBoundingClientRect();
    const timeAt = (clientX) => {
      const rect = plotArea();
      const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      return this.range[0] + fraction * (this.range[1] - this.range[0]);
    };
    this.root.addEventListener("pointermove", (event) => {
      if (event.pointerType === "mouse" || event.buttons) this.onCursor(timeAt(event.clientX));
    });
    this.root.addEventListener("pointerdown", (event) => this.onCursor(timeAt(event.clientX)));
    this.root.addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        onStep(event.key === "ArrowRight" ? 1 : -1);
      }
    });

    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(this.root);
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
      if (track.axis) this.drawAxis(context, width, height);
      else track.draw(context, width, height, this.xOf(width));
    }
    this.placeCursor(this.cursorTime);
  }

  drawAxis(context, width, height) {
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
 * maximum, with optional HPD bands, a position marker, and a clickable strip
 * of place fields for choosing the firing cell.
 */
export class DistributionChart {
  constructor(container, { positionBins, xLabel, plotHeight = 150, cellStrip = null, width = 640 }) {
    this.bins = positionBins;
    this.width = width;
    this.margin = { left: 12, right: 12, top: 10 };
    this.plotHeight = plotHeight;
    // Below the plot: two HPD bands, the axis, tick labels, and the axis title.
    this.axisBlock = 60;
    this.stripHeight = 34;
    this.cellStrip = cellStrip;
    const totalHeight =
      this.margin.top + plotHeight + this.axisBlock + (cellStrip ? this.stripHeight + 30 : 0);
    this.root = svg("svg", {
      class: "dist-chart",
      viewBox: `0 0 ${this.width} ${totalHeight}`,
      role: "img",
    });
    container.appendChild(this.root);
    this.xMin = positionBins[0];
    this.xMax = positionBins[positionBins.length - 1];

    this.layers = {
      bands: svg("g", {}, this.root),
      areas: svg("g", {}, this.root),
      marker: svg("g", {}, this.root),
      axis: svg("g", { class: "axis" }, this.root),
      cells: svg("g", {}, this.root),
    };
    this.drawAxis(xLabel);
  }

  x(value) {
    const { left, right } = this.margin;
    return left + ((value - this.xMin) / (this.xMax - this.xMin)) * (this.width - left - right);
  }

  drawAxis(xLabel) {
    const y = this.margin.top + this.plotHeight + 22;
    svg("line", { x1: this.x(this.xMin), x2: this.x(this.xMax), y1: y, y2: y }, this.layers.axis);
    for (const tick of niceTicks(this.xMin, this.xMax, 6)) {
      svg("line", { x1: this.x(tick), x2: this.x(tick), y1: y, y2: y + 4 }, this.layers.axis);
      const label = svg("text", { x: this.x(tick), y: y + 16, "text-anchor": "middle" });
      label.textContent = String(tick);
      this.layers.axis.appendChild(label);
    }
    const title = svg("text", { x: this.width / 2, y: y + 32, "text-anchor": "middle" });
    title.textContent = xLabel;
    this.layers.axis.appendChild(title);
  }

  curvePath(values, closed) {
    const max = Math.max(...values);
    const base = this.margin.top + this.plotHeight;
    const yOf = (v) => base - (max > 0 ? v / max : 0) * (this.plotHeight - 4);
    let d = "";
    values.forEach((v, i) => {
      d += `${i === 0 ? "M" : "L"}${this.x(this.bins[i]).toFixed(2)},${yOf(v).toFixed(2)}`;
    });
    if (closed) {
      d += `L${this.x(this.bins[this.bins.length - 1])},${base}L${this.x(this.bins[0])},${base}Z`;
    }
    return d;
  }

  /** series: [{values, color}], bands: [{mask, color}], marker: position | null */
  update({ series, bands = [], marker = null, selectedCell = null, cellCenters = null }) {
    for (const layer of ["bands", "areas", "marker"]) this.layers[layer].replaceChildren();
    for (const { values, color } of series) {
      svg("path", { d: this.curvePath(values, true), fill: color, "fill-opacity": 0.12 }, this.layers.areas);
      svg(
        "path",
        { d: this.curvePath(values, false), fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round" },
        this.layers.areas,
      );
    }
    const step = this.bins.length > 1 ? this.x(this.bins[1]) - this.x(this.bins[0]) : 1;
    bands.forEach(({ mask, color }, index) => {
      const y = this.margin.top + this.plotHeight + 4 + index * 7;
      mask.forEach((inside, i) => {
        if (!inside) return;
        svg(
          "rect",
          { x: this.x(this.bins[i]) - step / 2, y, width: step + 0.3, height: 5, fill: color },
          this.layers.bands,
        );
      });
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
    if (this.cellStrip && cellCenters) this.drawCells(selectedCell, cellCenters);
  }

  drawCells(selectedCell, cellCenters) {
    const { rates, selectable, onSelect } = this.cellStrip();
    this.layers.cells.replaceChildren();
    const top = this.margin.top + this.plotHeight + this.axisBlock + 22;
    const stripHeight = this.stripHeight;
    for (const cell of selectable) {
      const column = rates.map((row) => row[cell]);
      const max = Math.max(...column);
      let d = "";
      column.forEach((v, i) => {
        const y = top + stripHeight - (v / max) * stripHeight;
        d += `${i === 0 ? "M" : "L"}${this.x(this.bins[i]).toFixed(1)},${y.toFixed(1)}`;
      });
      const group = svg("g", { class: "cell-hit", tabindex: 0, role: "button" }, this.layers.cells);
      const title = svg("title", {}, group);
      title.textContent = `Cell ${cell + 1}: field centered at ${cellCenters[cell]}`;
      group.setAttribute("aria-label", title.textContent);
      group.setAttribute("aria-pressed", String(cell === selectedCell));
      const center = this.x(cellCenters[cell]);
      svg("rect", { x: center - 12, y: top - 4, width: 24, height: stripHeight + 8, fill: "transparent" }, group);
      svg(
        "path",
        {
          d,
          fill: "none",
          stroke: cell === selectedCell ? cssVar("--likelihood") : "#b5b5b1",
          "stroke-width": cell === selectedCell ? 2.5 : 1.2,
        },
        group,
      );
      const choose = () => onSelect(cell);
      group.addEventListener("click", choose);
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          choose();
        }
      });
    }
    const caption = svg("text", { x: this.margin.left, y: top - 8 });
    caption.textContent = "Place fields — click one to choose the cell that fired";
    this.layers.cells.appendChild(caption);
  }
}
