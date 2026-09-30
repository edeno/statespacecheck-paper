// The browser reads the explorer's binary containers exactly as the Python
// writer packed them. fixtures/container.bin is written by
// tests/test_site_explorer_export.py::container_fixture_bytes, which also
// checks that the committed file is current.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { readContainer } from "../js/container.js";

const file = readFileSync(new URL("fixtures/container.bin", import.meta.url));
const { meta, arrays } = readContainer(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));

test("the header's meta survives", () => {
  assert.deepEqual(meta, { example: "reference", values: [1, 2.5] });
});

test("every dtype decodes to its values and shape", () => {
  assert.deepEqual([...arrays.u1], [0, 7, 255]);
  assert.deepEqual([...arrays.u2], [1, 65535, 300, 2]);
  assert.deepEqual(arrays.u2.shape, [2, 2]);
  assert.deepEqual([...arrays.u4], [4_000_000_000, 5]);
  assert.deepEqual([...arrays.i4], [-1, 2 ** 31 - 1]);
  assert.deepEqual([...arrays.f4], [0.5, -2.25, Math.fround(1e-14)]);
  assert.deepEqual([...arrays.f8], [Math.PI]);
  assert.equal(arrays.empty.length, 0);
  assert.deepEqual(arrays.empty.shape, [0, 3]);
});

test("delta-filtered arrays are restored row by row, through wraparound", () => {
  assert.deepEqual([...arrays.delta_u1], [10, 12, 5, 250, 0, 0, 1, 255]);
  assert.deepEqual([...arrays.delta_u4], [3, 3, 9, 4_000_000_000]);
});

test("anything but a container is rejected", () => {
  assert.throws(() => readContainer(new ArrayBuffer(16)), /Not a recording-explorer container/);
});
