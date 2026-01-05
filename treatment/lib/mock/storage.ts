import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { Cart } from '../shopify/types';

// Storage file location
const STORAGE_DIR = join(process.cwd(), '.cart-storage');
const STORAGE_FILE = join(STORAGE_DIR, 'carts.json');

interface CartStorage {
  carts: Record<string, Cart>;
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
