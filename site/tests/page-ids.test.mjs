// Every element id the scripts look up must exist in the page: a renamed id
// otherwise fails only at runtime, when a player finds null.

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const siteDir = fileURLToPath(new URL("..", import.meta.url));
const jsDir = join(siteDir, "js");

test("every id the scripts query is defined in index.html", () => {
  const html = readFileSync(join(siteDir, "index.html"), "utf8");
  const pageIds = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));
  const missing = [];
  for (const file of readdirSync(jsDir).filter((name) => name.endsWith(".js"))) {
    const source = readFileSync(join(jsDir, file), "utf8");
    for (const [, id] of source.matchAll(/querySelector(?:All)?\("#([\w-]+)/g)) {
      if (!pageIds.has(id)) missing.push(`${file}: #${id}`);
    }
  }
  assert.deepEqual(missing, []);
});
