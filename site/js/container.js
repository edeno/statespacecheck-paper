// Reader for the binary containers of the recording explorer, written by
// statespacecheck_paper.site_explorer_export.encode_container: the ASCII magic
// "SSCX", a little-endian uint32 header length, a UTF-8 JSON header naming each
// array's dtype, shape, offset, and filter, then 8-byte-aligned arrays.

const TYPED = {
  u1: Uint8Array,
  u2: Uint16Array,
  u4: Uint32Array,
  i4: Int32Array,
  f4: Float32Array,
  f8: Float64Array,
};

/** Undo the "delta" filter: a running sum along the last axis, wrapping like the dtype. */
function undoDelta(values, shape) {
  const width = shape.length ? shape[shape.length - 1] : 0;
  for (let start = 0; start < values.length; start += width) {
    // Assigning to an unsigned typed array wraps the sum to its range.
    for (let i = start + 1; i < start + width; i += 1) values[i] += values[i - 1];
  }
  return values;
}

/**
 * Decode a container from an ArrayBuffer: {meta, arrays}, each array a typed
 * array (row-major) with its `shape` attached.
 */
export function readContainer(buffer) {
  const bytes = new Uint8Array(buffer);
  if (String.fromCharCode(...bytes.subarray(0, 4)) !== "SSCX") {
    throw new Error("Not a recording-explorer container");
  }
  const length = new DataView(buffer).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + length)));
  const arrays = {};
  for (const entry of header.arrays) {
    const Typed = TYPED[entry.dtype];
    if (!Typed) throw new Error(`Unknown dtype ${entry.dtype}`);
    const count = entry.shape.reduce((product, n) => product * n, 1);
    // Copy out of the shared buffer so filters and callers never alias it.
    const values = new Typed(buffer.slice(entry.offset, entry.offset + count * Typed.BYTES_PER_ELEMENT));
    if (entry.filter === "delta") undoDelta(values, entry.shape);
    else if (entry.filter) throw new Error(`Unknown filter ${entry.filter}`);
    values.shape = entry.shape;
    arrays[entry.name] = values;
  }
  return { meta: header.meta, arrays };
}

/** Fetch a gzipped container; the browser decompresses it (DecompressionStream). */
export async function loadContainer(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  if (!response.body || typeof DecompressionStream === "undefined") {
    throw new Error("This browser cannot read the compressed recording files");
  }
  const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
  return readContainer(await new Response(stream).arrayBuffer());
}
