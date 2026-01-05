import { cookies } from 'next/headers';
import {
  unstable_cacheTag as cacheTag,
  unstable_cacheLife as cacheLife
} from 'next/cache';
import { NextRequest, NextResponse } from 'next/server';
import { Cart, Collection, Menu, Page, Product } from '../shopify/types';
import { TAGS } from '../constants';
import { mockProducts, mockCollections, mockMenus, mockPages } from './data';
import { getStoredCart, setStoredCart, deleteStoredCart, clearAllCarts } from './storage';

function generateCartId(): string {
  return `cart_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

function createEmptyCart(id: string): Cart {
  return {
    id,
    checkoutUrl: '/checkout',
    cost: {
      subtotalAmount: { amount: '0.00', currencyCode: 'USD' },
      totalAmount: { amount: '0.00', currencyCode: 'USD' },
      totalTaxAmount: { amount: '0.00', currencyCode: 'USD' }
    },
    lines: [],
    totalQuantity: 0
  };
}

function recalculateCart(cart: Cart): Cart {
  let subtotal = 0;
  let totalQuantity = 0;

  for (const line of cart.lines) {
    subtotal += parseFloat(line.cost.totalAmount.amount);
    totalQuantity += line.quantity;
  }

  return {
    ...cart,
    cost: {
      subtotalAmount: { amount: subtotal.toFixed(2), currencyCode: 'USD' },
      totalAmount: { amount: subtotal.toFixed(2), currencyCode: 'USD' },
      totalTaxAmount: { amount: '0.00', currencyCode: 'USD' }
    },
    totalQuantity
  };
}

export async function createCart(): Promise<Cart> {
  const id = generateCartId();
  const cart = createEmptyCart(id);
  setStoredCart(id, cart);
  return cart;
}

export async function addToCart(
  lines: { merchandiseId: string; quantity: number }[]
): Promise<Cart> {
  const cookieStore = await cookies();
  let cartId = cookieStore.get('cartId')?.value;

  // Auto-create cart if it doesn't exist
  let cart = cartId ? getStoredCart(cartId) : undefined;
  if (!cart) {
    const newCart = await createCart();
    cartId = newCart.id!;
    cart = newCart;
    cookieStore.set('cartId', cartId);
  }

  for (const line of lines) {
    // Find the product and variant
    let foundProduct: Product | undefined;
    let foundVariant: Product['variants'][0] | undefined;

    for (const product of mockProducts) {
      const variant = product.variants.find(v => v.id === line.merchandiseId);
      if (variant) {
        foundProduct = product;
        foundVariant = variant;
        break;
      }
    }

    if (!foundProduct || !foundVariant) {
      continue;
    }

    // Check if line already exists
    const existingLineIndex = cart.lines.findIndex(
      l => l.merchandise.id === line.merchandiseId
    );

    if (existingLineIndex >= 0) {
      const existingLine = cart!.lines[existingLineIndex]!;
      existingLine.quantity += line.quantity;
      existingLine.cost.totalAmount.amount = (
        parseFloat(foundVariant.price.amount) * existingLine.quantity
      ).toFixed(2);
    } else {
      cart.lines.push({
        id: `line_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
        quantity: line.quantity,
        cost: {
          totalAmount: {
            amount: (parseFloat(foundVariant.price.amount) * line.quantity).toFixed(2),
            currencyCode: 'USD'
          }
        },
        merchandise: {
          id: foundVariant.id,
          title: foundVariant.title,
          selectedOptions: foundVariant.selectedOptions,
          product: {
            id: foundProduct.id,
            handle: foundProduct.handle,
            title: foundProduct.title,
            featuredImage: foundProduct.featuredImage
          }
        }
      });
    }
  }

  const updatedCart = recalculateCart(cart);
  setStoredCart(cartId!, updatedCart);
  return updatedCart;
}

export async function removeFromCart(lineIds: string[]): Promise<Cart> {
  const cartId = (await cookies()).get('cartId')?.value;

  if (!cartId) {
    throw new Error('Cart not found');
  }

  const cart = getStoredCart(cartId);
  if (!cart) {
    throw new Error('Cart not found');
  }

  cart.lines = cart.lines.filter(line => !lineIds.includes(line.id!));

  const updatedCart = recalculateCart(cart);
  setStoredCart(cartId, updatedCart);
  return updatedCart;
}

export async function updateCart(
  lines: { id: string; merchandiseId: string; quantity: number }[]
): Promise<Cart> {
  const cartId = (await cookies()).get('cartId')?.value;

  if (!cartId) {
    throw new Error('Cart not found');
  }

  const cart = getStoredCart(cartId);
  if (!cart) {
    throw new Error('Cart not found');
  }

  for (const line of lines) {
    const existingLine = cart.lines.find(l => l.id === line.id);
    if (existingLine) {
      if (line.quantity === 0) {
        cart.lines = cart.lines.filter(l => l.id !== line.id);
      } else {
        existingLine.quantity = line.quantity;
        // Find price for the variant
        for (const product of mockProducts) {
          const variant = product.variants.find(v => v.id === line.merchandiseId);
          if (variant) {
            existingLine.cost.totalAmount.amount = (
              parseFloat(variant.price.amount) * line.quantity
            ).toFixed(2);
            break;
          }
        }
      }
    }
  }

  const updatedCart = recalculateCart(cart);
  setStoredCart(cartId, updatedCart);
  return updatedCart;
}

export async function getCart(): Promise<Cart | undefined> {
  const cartId = (await cookies()).get('cartId')?.value;

  if (!cartId) {
    return undefined;
  }

  return getStoredCart(cartId);
}

export async function getCollection(handle: string): Promise<Collection | undefined> {
  return mockCollections.find(c => c.handle === handle);
}

export async function getCollectionProducts({
  collection,
  reverse,
  sortKey
}: {
  collection: string;
  reverse?: boolean;
  sortKey?: string;
}): Promise<Product[]> {
  let products = [...mockProducts];

  // Filter by collection tag if not "all"
  if (collection && collection !== '') {
    products = products.filter(p =>
      p.tags.some(tag =>
        tag.toLowerCase().includes(collection.toLowerCase()) ||
        collection.toLowerCase().includes(tag.toLowerCase())
      )
    );
  }

  // Sort products
  if (sortKey) {
    products.sort((a, b) => {
      switch (sortKey) {
        case 'PRICE':
          return parseFloat(a.priceRange.minVariantPrice.amount) -
                 parseFloat(b.priceRange.minVariantPrice.amount);
        case 'CREATED_AT':
        case 'CREATED':
          return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
        case 'TITLE':
          return a.title.localeCompare(b.title);
        default:
          return 0;
      }
    });
  }

  if (reverse) {
    products.reverse();
  }

  return products;
}

export async function getCollections(): Promise<Collection[]> {
  return mockCollections;
}

export async function getMenu(handle: string): Promise<Menu[]> {
  return mockMenus[handle] || [];
}

export async function getPage(handle: string): Promise<Page> {
  const page = mockPages.find(p => p.handle === handle);
  if (!page) {
    throw new Error(`Page not found: ${handle}`);
  }
  return page;
}

export async function getPages(): Promise<Page[]> {
  return mockPages;
}

export async function getProduct(handle: string): Promise<Product | undefined> {
  return mockProducts.find(p => p.handle === handle);
}

export async function getProductRecommendations(productId: string): Promise<Product[]> {
  // Return 4 random products excluding the current one
  const filtered = mockProducts.filter(p => p.id !== productId);
  const shuffled = filtered.sort(() => 0.5 - Math.random());
  return shuffled.slice(0, 4);
}

export async function getProducts({
  query,
  reverse,
  sortKey
}: {
  query?: string;
  reverse?: boolean;
  sortKey?: string;
}): Promise<Product[]> {
  let products = [...mockProducts];

  // Filter by search query
  if (query) {
    const lowerQuery = query.toLowerCase();
    products = products.filter(p =>
      p.title.toLowerCase().includes(lowerQuery) ||
      p.description.toLowerCase().includes(lowerQuery) ||
      p.tags.some(t => t.toLowerCase().includes(lowerQuery))
    );
  }

  // Sort products
  if (sortKey) {
    products.sort((a, b) => {
      switch (sortKey) {
        case 'PRICE':
          return parseFloat(a.priceRange.minVariantPrice.amount) -
                 parseFloat(b.priceRange.minVariantPrice.amount);
        case 'CREATED_AT':
        case 'CREATED':
          return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
        case 'TITLE':
          return a.title.localeCompare(b.title);
        default:
          return 0;
      }
    });
  }

  if (reverse) {
    products.reverse();
  }

  return products;
}

export async function revalidate(req: NextRequest): Promise<NextResponse> {
  // Mock provider doesn't need revalidation
  return NextResponse.json({ status: 200, revalidated: true, now: Date.now() });
}

// Export clearAllCarts for the reset endpoint
export { clearAllCarts };
