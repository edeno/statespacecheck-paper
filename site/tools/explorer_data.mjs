// Put the session explorer's spike index and recording blocks into the built
// site. They live in one archive, published as a release asset rather than
// committed; the committed overview names it and records its SHA-256.
// Usage: node tools/explorer_data.mjs <overview.json> <archive dir> <built data dir>
// Uses <archive dir>/<name> when its checksum matches (the export script writes
// it there), and otherwise downloads it there from the overview's URL.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const [overviewPath, archiveDir, dataDir] = process.argv.slice(2);
const { archive } = JSON.parse(readFileSync(overviewPath, "utf8"));
const path = join(archiveDir, archive.name);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

if (!existsSync(path) || sha256(readFileSync(path)) !== archive.sha256) {
  console.log(`Downloading ${archive.url} (${(archive.bytes / 1e6).toFixed(1)} MB)`);
  const response = await fetch(archive.url);
  if (!response.ok) {
    throw new Error(
      `${archive.url}: HTTP ${response.status}. Publish the archive (scripts/export_recording_explorer.py prints the command) or place it in ${archiveDir}.`,
    );
  }
  const bytes = Buffer.from(await response.arrayBuffer());
  if (sha256(bytes) !== archive.sha256) {
    throw new Error(`${archive.url} does not match the SHA-256 in ${overviewPath}`);
  }
  mkdirSync(archiveDir, { recursive: true });
  writeFileSync(path, bytes);
}

rmSync(join(dataDir, "explorer"), { recursive: true, force: true });
execFileSync("tar", ["-xf", path, "-C", dataDir]);
console.log(`Extracted ${archive.name} into ${dataDir}`);
