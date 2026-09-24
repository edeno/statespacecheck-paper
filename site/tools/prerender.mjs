// Fill the page's reported-value placeholders at build time, so the numbers
// read correctly without JavaScript (and for link previews and crawlers).
// Usage: node tools/prerender.mjs <built index.html> <manifest.json>
// The page's own JavaScript fills the same placeholders from the same manifest.

import { readFileSync, writeFileSync } from "node:fs";

import { formatMacro } from "../js/data.js";

const [htmlPath, manifestPath] = process.argv.slice(2);
const { macros } = JSON.parse(readFileSync(manifestPath, "utf8"));
const html = readFileSync(htmlPath, "utf8");

const unknown = new Set();
const filled = html.replace(
  /<span data-macro="([A-Za-z]+)"( data-format="count")?><\/span>/g,
  (_, name, count) => {
    if (!(name in macros)) {
      unknown.add(name);
      return _;
    }
    const value = formatMacro(macros[name], count ? "count" : undefined);
    return `<span data-macro="${name}"${count ?? ""}>${value}</span>`;
  },
);
if (unknown.size) {
  console.error(`Unknown macros: ${[...unknown].join(", ")}`);
  process.exit(1);
}
const remaining = filled.match(/<span data-macro="[^"]*"[^>]*><\/span>/g);
if (remaining) {
  console.error(`Unfilled placeholders: ${remaining.join(", ")}`);
  process.exit(1);
}
writeFileSync(htmlPath, filled);
