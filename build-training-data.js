/**
 * Build Naat vs Explanation training data from a local audio file
 *
 * Replaces the old Appwrite-backed exporter. Takes a downloaded audio file and a
 * list of explanation (speech) segments, derives the naat segments as the inverse,
 * and splits everything into labeled 5-second 16kHz mono WAV chunks.
 *
 * Output structure:
 *   training-data/
 *   ├── naat/          # 5-sec .wav chunks of naat recitation
 *   ├── explanation/   # 5-sec .wav chunks of explanation speech
 *   └── manifest.json  # metadata about all chunks
 *
 * Usage:
 *   node build-training-data.js "<source-audio-or-video>" [endSeconds]
 *
 * Example:
 *   node build-training-data.js .tmp/om7nbXHFjMk.wav 246
 *   (uses EXPLANATION_SEGMENTS below for labeling)
 */

const { execFileSync } = require("child_process");
const { existsSync, mkdirSync, writeFileSync, unlinkSync, rmSync, readdirSync } = require("fs");
const { join } = require("path");

// ── Config ────────────────────────────────────────────────────
const CHUNK_DURATION = 5; // seconds per training chunk
const SAMPLE_RATE = 16000; // 16kHz for Wav2Vec2
const OUTPUT_DIR = join(__dirname, "training-data");
const TEMP_DIR = join(__dirname, ".tmp");
const NAAT_DIR = join(OUTPUT_DIR, "naat");
const EXPLANATION_DIR = join(OUTPUT_DIR, "explanation");

// Chunks quieter than this RMS are treated as silence and EXCLUDED from training.
// Labeling quiet audio as either class teaches the model "quiet = explanation",
// which makes inference cut quiet naat passages. Keep it out entirely.
// NaNAT chunks here sit at RMS >= ~0.05; quiet speech drops below ~0.03.
const SILENCE_RMS_THRESHOLD = 0.03;

const SOURCE_AUDIO = process.argv[2];
const END_SECONDS = parseFloat(process.argv[3]);

// Explanation (speech) segments in seconds. Naat = inverse of these within [0, end].
const EXPLANATION_SEGMENTS = [
  { start: 58, end: 157 },    // 0:58 - 2:37
  { start: 189, end: 214 },   // 3:09 - 3:34
  { start: 242, end: 246 },   // 4:02 - 4:06
];

// Source id used in chunk filenames
const SOURCE_ID = "om7nbXHFjMk";

// ── Helpers ───────────────────────────────────────────────────

function ensureDirs() {
  if (existsSync(OUTPUT_DIR)) {
    rmSync(OUTPUT_DIR, { recursive: true, force: true });
  }
  [OUTPUT_DIR, TEMP_DIR, NAAT_DIR, EXPLANATION_DIR].forEach((dir) => mkdirSync(dir, { recursive: true }));
}

function ffprobeDuration(filePath) {
  const out = execFileSync("ffprobe", [
    "-v", "error",
    "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1",
    filePath,
  ], { encoding: "utf-8" });
  return parseFloat(out.trim()) || 0;
}

/**
 * Build labeled ranges from explanation segments + total duration.
 */
function buildLabeledRanges(explanationRanges, totalDuration) {
  const sorted = [...explanationRanges].sort((a, b) => a.start - b.start);
  const naatRanges = [];
  let cursor = 0;

  for (const seg of sorted) {
    if (cursor < seg.start) naatRanges.push({ start: cursor, end: seg.start });
    cursor = Math.max(cursor, seg.end);
  }
  if (cursor < totalDuration) naatRanges.push({ start: cursor, end: totalDuration });

  return { naat: naatRanges, explanation: sorted.map((s) => ({ start: s.start, end: s.end })) };
}

/**
 * Split a time range into fixed-size chunks.
 * Drops the last chunk if it's less than half the chunk duration.
 */
function splitIntoChunks(ranges, chunkDuration) {
  const chunks = [];
  for (const range of ranges) {
    let t = range.start;
    while (t + chunkDuration <= range.end) {
      chunks.push({ start: t, end: t + chunkDuration });
      t += chunkDuration;
    }
    const remaining = range.end - t;
    if (remaining >= chunkDuration / 2) {
      chunks.push({ start: t, end: range.end });
    }
  }
  return chunks;
}

/**
 * Extract a single chunk from audio as 16kHz mono WAV.
 */
function extractChunk(inputPath, start, duration, outputPath) {
  execFileSync("ffmpeg", [
    "-y", "-hide_banner", "-loglevel", "error",
    "-ss", String(start),
    "-i", inputPath,
    "-t", String(duration),
    "-ar", String(SAMPLE_RATE),
    "-ac", "1",
    "-c:a", "pcm_s16le",
    outputPath,
  ]);
}

/**
 * Compute RMS (root mean square) of a 16-bit PCM mono WAV file.
 */
function rmsOfWavFile(filePath) {
  const buf = require("fs").readFileSync(filePath);
  let off = 12;
  while (off < buf.length && buf.readUInt32LE(off) !== 0x61746164) {
    const sz = buf.readUInt32LE(off + 4);
    off += 8 + sz + (sz % 2);
  }
  const dataStart = off + 8;
  const n = Math.floor((buf.length - dataStart) / 2);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const s = buf.readInt16LE(dataStart + i * 2) / 32768;
    sum += s * s;
  }
  return Math.sqrt(sum / Math.max(1, n));
}

function extractAll(labelDir, label, inputPath, chunks) {
  const kept = [];
  let skipped = 0;
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const suffix = label === "naat" ? "naat" : "expl";
    const filename = `${SOURCE_ID}_${suffix}_${String(kept.length).padStart(3, "0")}.wav`;
    const outPath = join(labelDir, filename);
    extractChunk(inputPath, chunk.start, chunk.end - chunk.start, outPath);

    const rms = rmsOfWavFile(outPath);
    if (rms < SILENCE_RMS_THRESHOLD) {
      try { unlinkSync(outPath); } catch { /* ignore */ }
      skipped++;
      console.log(`     skipping ${filename} (RMS ${rms.toFixed(4)} < ${SILENCE_RMS_THRESHOLD})`);
      continue;
    }
    kept.push({ ...chunk, filename });
  }
  return { kept, skipped };
}

// ── Main ──────────────────────────────────────────────────────

function main() {
  if (!SOURCE_AUDIO) {
    console.error("Usage: node build-training-data.js <source-audio> [endSeconds]");
    process.exit(1);
  }
  if (!existsSync(SOURCE_AUDIO)) {
    console.error(`❌ Source audio not found: ${SOURCE_AUDIO}`);
    process.exit(1);
  }

  ensureDirs();

  // Trim/resample source to 16kHz mono, capped at END_SECONDS.
  const trimmedPath = join(TEMP_DIR, "original.wav");
  console.log(`🔧 Normalizing ${SOURCE_AUDIO} to 16kHz mono...`);
  const trimArgs = [
    "-y", "-hide_banner", "-loglevel", "error",
    "-i", SOURCE_AUDIO,
    "-ar", String(SAMPLE_RATE),
    "-ac", "1",
  ];
  if (END_SECONDS) trimArgs.push("-t", String(END_SECONDS));
  trimArgs.push(trimmedPath);
  execFileSync("ffmpeg", trimArgs);

  const totalDuration = ffprobeDuration(trimmedPath);
  console.log(`   Duration used: ${totalDuration.toFixed(1)}s\n`);

  const { naat: naatRanges, explanation: explanationRanges } =
    buildLabeledRanges(EXPLANATION_SEGMENTS, totalDuration);
  console.log(`  Naat ranges: ${naatRanges.map((r) => `${r.start}-${r.end}`).join(", ")}`);
  console.log(`  Explanation ranges: ${explanationRanges.map((r) => `${r.start}-${r.end}`).join(", ")}\n`);

  const rawNaatChunks = splitIntoChunks(naatRanges, CHUNK_DURATION);
  const rawExplanationChunks = splitIntoChunks(explanationRanges, CHUNK_DURATION);
  console.log(`  Chunks: ${rawNaatChunks.length} naat, ${rawExplanationChunks.length} explanation`);

  console.log("\n🎧 Extracting naat chunks...");
  const { kept: naatChunks, skipped: skippedNaat } = extractAll(NAAT_DIR, "naat", trimmedPath, rawNaatChunks);
  console.log("🎙️  Extracting explanation chunks...");
  const { kept: explanationChunks, skipped: skippedExpl } = extractAll(EXPLANATION_DIR, "explanation", trimmedPath, rawExplanationChunks);

  // ── Manifest ─────────────────────────────────────────────
  const manifest = [];
  naatChunks.forEach((c) => manifest.push({
    file: `naat/${c.filename}`,
    label: "naat",
    source: SOURCE_ID,
    start: c.start,
    end: c.end,
  }));
  explanationChunks.forEach((c) => manifest.push({
    file: `explanation/${c.filename}`,
    label: "explanation",
    source: SOURCE_ID,
    start: c.start,
    end: c.end,
  }));
  writeFileSync(join(OUTPUT_DIR, "manifest.json"), JSON.stringify(manifest, null, 2));

  // ── Cleanup trimmed temp file ────────────────────────────
  try { unlinkSync(trimmedPath); } catch { /* ignore */ }
  try {
    for (const f of readdirSync(TEMP_DIR)) {
      if (f.startsWith("om7nbXHFjMk")) { try { unlinkSync(join(TEMP_DIR, f)); } catch { /* ignore */ } }
    }
  } catch { /* ignore */ }

  // ── Summary ─────────────────────────────────────────────
  const total = naatChunks.length + explanationChunks.length;
  console.log("\n════════════════════════════════════════");
  console.log("  📊 Build Summary");
  console.log("════════════════════════════════════════");
  console.log(`  Total chunks:       ${total}`);
  console.log(`  Naat chunks:        ${naatChunks.length}`);
  console.log(`  Explanation chunks: ${explanationChunks.length}`);
  if (skippedNaat || skippedExpl) {
    console.log(`  Silence dropped:    ${skippedNaat + skippedExpl} (naat ${skippedNaat}, explanation ${skippedExpl})`);
  }
  console.log(`  Class balance:      ${((naatChunks.length / total) * 100).toFixed(1)}% naat / ${((explanationChunks.length / total) * 100).toFixed(1)}% explanation`);
  console.log("════════════════════════════════════════\n");
}

main();