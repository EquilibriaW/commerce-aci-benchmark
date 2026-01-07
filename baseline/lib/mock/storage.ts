import { readFileSync, writeFileSync, existsSync, mkdirSync, unlinkSync, renameSync, readdirSync } from 'fs';
import { join } from 'path';
import { Cart } from '../shopify/types';

/**
 * Per-session file storage for benchmark parallelism.
 *
 * Uses individual files per cart/order to eliminate cross-session contention:
 *   .cart-storage/carts/{cartId}.json
 *   .cart-storage/orders/{sessionId}.json
 *
 * Writes are atomic: write to temp file, then rename to final path.
 */

const STORAGE_DIR = join(process.cwd(), '.cart-storage');
const CARTS_DIR = join(STORAGE_DIR, 'carts');
const ORDERS_DIR = join(STORAGE_DIR, 'orders');

// Order structure for completed checkouts
export interface CompletedOrder {
  id: string;
  customer: {
    name: string;
    email: string;
  };
  items: Array<{
    id: string;
    slug: string;
    title: string;
    variant: string;
    quantity: number;
    unit_price_cents: number;
    line_total_cents: number;
  }>;
  total_items: number;
  total_price_cents: number;
  currency: string;
  completed_at: string;
}

function ensureDir(dir: string): void {
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
}

function ensureStorageDirs(): void {
  ensureDir(CARTS_DIR);
  ensureDir(ORDERS_DIR);
}

/**
 * Atomic write: write to temp file then rename.
 * This prevents partial writes from corrupting data during concurrent access.
 */
function atomicWriteJSON(filePath: string, data: unknown): void {
  const tempPath = `${filePath}.${process.pid}.tmp`;
  try {
    writeFileSync(tempPath, JSON.stringify(data, null, 2), 'utf-8');
    renameSync(tempPath, filePath);
  } catch (error) {
    // Clean up temp file on error
    try { unlinkSync(tempPath); } catch { /* ignore */ }
    throw error;
  }
}

function readJSON<T>(filePath: string): T | undefined {
  if (!existsSync(filePath)) {
    return undefined;
  }
  try {
    const data = readFileSync(filePath, 'utf-8');
    return JSON.parse(data) as T;
  } catch (error) {
    console.error(`Error reading ${filePath}:`, error);
    return undefined;
  }
}

function safeFilename(id: string): string {
  // Sanitize ID to be filesystem-safe (replace unsafe chars with underscore)
  return id.replace(/[^a-zA-Z0-9_-]/g, '_');
}

// --- CART STORAGE ---

function cartPath(cartId: string): string {
  return join(CARTS_DIR, `${safeFilename(cartId)}.json`);
}

export function getStoredCart(cartId: string): Cart | undefined {
  ensureStorageDirs();
  return readJSON<Cart>(cartPath(cartId));
}

export function setStoredCart(cartId: string, cart: Cart): void {
  ensureStorageDirs();
  atomicWriteJSON(cartPath(cartId), cart);
}

export function deleteStoredCart(cartId: string): void {
  ensureStorageDirs();
  const path = cartPath(cartId);
  if (existsSync(path)) {
    try {
      unlinkSync(path);
    } catch (error) {
      console.error(`Error deleting cart ${cartId}:`, error);
    }
  }
}

export function clearAllCarts(): void {
  ensureStorageDirs();
  try {
    const files = readdirSync(CARTS_DIR);
    for (const file of files) {
      if (file.endsWith('.json')) {
        unlinkSync(join(CARTS_DIR, file));
      }
    }
  } catch (error) {
    console.error('Error clearing carts:', error);
  }
}

export function getAllCartIds(): string[] {
  ensureStorageDirs();
  try {
    const files = readdirSync(CARTS_DIR);
    return files
      .filter(f => f.endsWith('.json'))
      .map(f => f.replace('.json', ''));
  } catch {
    return [];
  }
}

// --- ORDER STORAGE ---

function orderPath(sessionId: string): string {
  return join(ORDERS_DIR, `${safeFilename(sessionId)}.json`);
}

export function saveCompletedOrder(sessionId: string, order: CompletedOrder): void {
  ensureStorageDirs();
  atomicWriteJSON(orderPath(sessionId), order);
}

export function getCompletedOrder(sessionId: string): CompletedOrder | undefined {
  ensureStorageDirs();
  return readJSON<CompletedOrder>(orderPath(sessionId));
}

export function deleteCompletedOrder(sessionId: string): void {
  ensureStorageDirs();
  const path = orderPath(sessionId);
  if (existsSync(path)) {
    try {
      unlinkSync(path);
    } catch (error) {
      console.error(`Error deleting order ${sessionId}:`, error);
    }
  }
}

export function clearAllOrders(): void {
  ensureStorageDirs();
  try {
    const files = readdirSync(ORDERS_DIR);
    for (const file of files) {
      if (file.endsWith('.json')) {
        unlinkSync(join(ORDERS_DIR, file));
      }
    }
  } catch (error) {
    console.error('Error clearing orders:', error);
  }
}

export function getLastCompletedOrder(): CompletedOrder | undefined {
  ensureStorageDirs();
  try {
    const files = readdirSync(ORDERS_DIR);
    let latestOrder: CompletedOrder | undefined;
    let latestTime = 0;

    for (const file of files) {
      if (!file.endsWith('.json')) continue;
      const order = readJSON<CompletedOrder>(join(ORDERS_DIR, file));
      if (order?.completed_at) {
        const time = new Date(order.completed_at).getTime();
        if (time > latestTime) {
          latestTime = time;
          latestOrder = order;
        }
      }
    }
    return latestOrder;
  } catch {
    return undefined;
  }
}
