import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { Cart } from '../shopify/types';

// Storage file location
const STORAGE_DIR = join(process.cwd(), '.cart-storage');
const STORAGE_FILE = join(STORAGE_DIR, 'carts.json');
const ORDERS_FILE = join(STORAGE_DIR, 'orders.json');

interface CartStorage {
  carts: Record<string, Cart>;
  lastUpdated: string;
}

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

interface OrderStorage {
  orders: Record<string, CompletedOrder>;  // keyed by session/cart ID
  lastUpdated: string;
}

function ensureStorageDir(): void {
  if (!existsSync(STORAGE_DIR)) {
    mkdirSync(STORAGE_DIR, { recursive: true });
  }
}

function readStorage(): CartStorage {
  ensureStorageDir();

  if (!existsSync(STORAGE_FILE)) {
    return { carts: {}, lastUpdated: new Date().toISOString() };
  }

  try {
    const data = readFileSync(STORAGE_FILE, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Error reading cart storage:', error);
    return { carts: {}, lastUpdated: new Date().toISOString() };
  }
}

function writeStorage(storage: CartStorage): void {
  ensureStorageDir();

  try {
    storage.lastUpdated = new Date().toISOString();
    writeFileSync(STORAGE_FILE, JSON.stringify(storage, null, 2), 'utf-8');
  } catch (error) {
    console.error('Error writing cart storage:', error);
  }
}

export function getStoredCart(cartId: string): Cart | undefined {
  const storage = readStorage();
  return storage.carts[cartId];
}

export function setStoredCart(cartId: string, cart: Cart): void {
  const storage = readStorage();
  storage.carts[cartId] = cart;
  writeStorage(storage);
}

export function deleteStoredCart(cartId: string): void {
  const storage = readStorage();
  delete storage.carts[cartId];
  writeStorage(storage);
}

export function clearAllCarts(): void {
  writeStorage({ carts: {}, lastUpdated: new Date().toISOString() });
}

export function getAllCartIds(): string[] {
  const storage = readStorage();
  return Object.keys(storage.carts);
}

// --- ORDER STORAGE ---

function readOrderStorage(): OrderStorage {
  ensureStorageDir();

  if (!existsSync(ORDERS_FILE)) {
    return { orders: {}, lastUpdated: new Date().toISOString() };
  }

  try {
    const data = readFileSync(ORDERS_FILE, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Error reading order storage:', error);
    return { orders: {}, lastUpdated: new Date().toISOString() };
  }
}

function writeOrderStorage(storage: OrderStorage): void {
  ensureStorageDir();

  try {
    storage.lastUpdated = new Date().toISOString();
    writeFileSync(ORDERS_FILE, JSON.stringify(storage, null, 2), 'utf-8');
  } catch (error) {
    console.error('Error writing order storage:', error);
  }
}

export function saveCompletedOrder(sessionId: string, order: CompletedOrder): void {
  const storage = readOrderStorage();
  storage.orders[sessionId] = order;
  writeOrderStorage(storage);
}

export function getCompletedOrder(sessionId: string): CompletedOrder | undefined {
  const storage = readOrderStorage();
  return storage.orders[sessionId];
}

export function clearAllOrders(): void {
  writeOrderStorage({ orders: {}, lastUpdated: new Date().toISOString() });
}
