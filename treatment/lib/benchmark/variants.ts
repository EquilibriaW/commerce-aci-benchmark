import { cookies } from 'next/headers';

export const DEFAULT_VARIANT_SEED = 0;
export const DEFAULT_VARIANT_LEVEL = 0;
export const MAX_VARIANT_LEVEL = 3;

export function parseVariantSeed(value?: string | null): number {
  const parsed = Number.parseInt(value ?? '', 10);
  if (Number.isNaN(parsed)) {
    return DEFAULT_VARIANT_SEED;
  }
  return parsed;
}

export function parseVariantLevel(value?: string | null): number {
  const parsed = Number.parseInt(value ?? '', 10);
  if (Number.isNaN(parsed)) {
    return DEFAULT_VARIANT_LEVEL;
  }
  return Math.max(0, Math.min(MAX_VARIANT_LEVEL, parsed));
}

export function getVariantFromCookies(): { seed: number; level: number } {
  const store = cookies();
  return {
    seed: parseVariantSeed(store.get('variantSeed')?.value),
    level: parseVariantLevel(store.get('variantLevel')?.value)
  };
}

export function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffle<T>(items: T[], rng: () => number): T[] {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
