// Stand-in parser. Real xlsx parsing comes later — for now we generate a
// plausible batch of samples so the table looks real during preview.
// The seed is derived from file size so the same file shows the same rows.

import type { BatchSample, CompositionRequest } from "./types";

interface ElementSpec {
  key: keyof CompositionRequest;
  /** [low, high] sampled uniformly. */
  range: [number, number];
  decimals: number;
  /** Probability that an optional element is *blank* on a given row. */
  blankP?: number;
}

// Ranges drawn from the labeled training distribution stats so the values
// scan as real Jominy data.
const SPECS: ElementSpec[] = [
  { key: "C",  range: [0.10, 0.45], decimals: 3 },
  { key: "Si", range: [0.15, 0.45], decimals: 3 },
  { key: "Mn", range: [0.40, 1.55], decimals: 3 },
  { key: "P",  range: [0.005, 0.025], decimals: 3 },
  { key: "S",  range: [0.005, 0.030], decimals: 3 },
  { key: "Cu", range: [0.02, 0.20], decimals: 3 },
  { key: "Ni", range: [0.02, 0.30], decimals: 3 },
  { key: "Cr", range: [0.20, 1.40], decimals: 3 },
  { key: "V",  range: [0.001, 0.030], decimals: 3, blankP: 0.30 },
  { key: "Ti", range: [0.001, 0.020], decimals: 3, blankP: 0.30 },
  { key: "W",  range: [0.001, 0.020], decimals: 3, blankP: 0.55 },
  { key: "Al", range: [0.005, 0.045], decimals: 3, blankP: 0.20 },
  { key: "B",  range: [0.0005, 0.0040], decimals: 4, blankP: 0.45 },
];

/** Mulberry32 — small deterministic PRNG seeded by an integer. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function round(value: number, decimals: number): number {
  const m = 10 ** decimals;
  return Math.round(value * m) / m;
}

function makeId(rand: () => number, index: number): string {
  // Mix of P-pattern heat numbers and plain numerics, like the real data.
  const r = rand();
  if (r < 0.7) {
    const num = Math.floor(rand() * 90_000_000) + 10_000_000;
    const suffix = rand() < 0.15 ? (rand() < 0.5 ? "-H" : "-Z") : "";
    return `P${num}${suffix}`;
  }
  return String(Math.floor(rand() * 9_000_000) + 1_000_000 + index);
}

function makeRow(rand: () => number, index: number): BatchSample {
  const composition = {} as CompositionRequest;
  for (const spec of SPECS) {
    const blank = spec.blankP !== undefined && rand() < spec.blankP;
    const slot = composition as unknown as Record<string, number | null>;
    if (blank) {
      // Optional → null, required can't be blank
      slot[spec.key] = null;
      continue;
    }
    const [lo, hi] = spec.range;
    slot[spec.key] = round(lo + rand() * (hi - lo), spec.decimals);
  }
  return {
    id: makeId(rand, index),
    composition,
    prediction: null,
    error: null,
    loading: true,
  };
}

/**
 * Mock parser. Returns ~8–14 samples seeded by file size so the same drop
 * produces the same data while the user is iterating on the design.
 *
 * Replace with the real `xlsx`-based parser later — the rest of the page
 * doesn't care where the samples come from.
 */
export async function parseXlsxMock(file: File): Promise<BatchSample[]> {
  // Tiny artificial delay to make the loading state visible.
  await new Promise((r) => setTimeout(r, 280));
  const seed = (file.size * 2654435761) ^ file.name.length;
  const rand = rng(seed);
  const count = 8 + Math.floor(rand() * 7); // 8..14
  return Array.from({ length: count }, (_, i) => makeRow(rand, i));
}
