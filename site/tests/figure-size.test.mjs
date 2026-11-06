// The page declares Figure 1's pixel size so the browser reserves its space
// before the image loads. Regenerating the figure can change that size, so the
// declared size must be the committed PNG's.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const siteDir = fileURLToPath(new URL("..", import.meta.url));
// The file site/Makefile copies to figures/figure01.png.
const figurePath = join(siteDir, "..", "manuscript", "figures", "main", "figure01.png");

test("the Figure 1 image declares the PNG's pixel size", () => {
  const html = readFileSync(join(siteDir, "index.html"), "utf8");
  const [tag] = html.match(/<img\b[^>]*\bsrc="figures\/figure01\.png"[^>]*>/) ?? [];
  assert.ok(tag, "index.html has an <img> for figures/figure01.png");
  const declared = {
    width: Number(tag.match(/\swidth="(\d+)"/)?.[1]),
    height: Number(tag.match(/\sheight="(\d+)"/)?.[1]),
  };
  // A PNG's IHDR chunk holds its width and height as big-endian uint32 at bytes 16 and 20.
  const png = readFileSync(figurePath);
  assert.deepEqual(declared, { width: png.readUInt32BE(16), height: png.readUInt32BE(20) });
});
