export const runtime = 'nodejs'; // CRITICAL: Ensures consistent server instance

import { getCart, getCompletedOrder } from 'lib/shopify';
import { NextRequest, NextResponse } from 'next/server';

const BENCHMARK_SECRET = process.env.BENCHMARK_SECRET;

// Robust string-to-cents conversion (avoids float precision issues)
function toCents(amount: string): number {
  const [dollars, cents = '00'] = amount.split('.');
  return parseInt(dollars, 10) * 100 + parseInt(cents.padEnd(2, '0').slice(0, 2), 10);
}

export async function GET(request: NextRequest) {
  // Fail closed - require secret
  if (!BENCHMARK_SECRET) {
    return NextResponse.json(
      { error: 'Server misconfigured', message: 'BENCHMARK_SECRET not set' },
      { status: 500, headers: { 'Cache-Control': 'no-store' } }
    );
  }

  // Security check - require benchmark secret header
  const providedSecret = request.headers.get('X-Benchmark-Secret');
  if (providedSecret !== BENCHMARK_SECRET) {
    return NextResponse.json(
      { error: 'Forbidden', message: 'Invalid or missing X-Benchmark-Secret header' },
      { status: 403, headers: { 'Cache-Control': 'no-store' } }
    );
  }

  try {
    const cart = await getCart();

    // Get the cartId from cookies to look up any completed order
    const cartIdCookie = request.cookies.get('cartId');
    const cartId = cartIdCookie?.value;
    const completedOrder = cartId ? getCompletedOrder(cartId) : undefined;

    const state = {
      cart: cart ? {
        id: cart.id,
        items: cart.lines.map(line => ({
          id: line.merchandise.id,
          slug: line.merchandise.product.handle,
          title: line.merchandise.product.title,
          variant: line.merchandise.title,
          quantity: line.quantity,
          unit_price_cents: toCents(line.merchandise.price.amount),
          line_total_cents: toCents(line.cost.totalAmount.amount)
        })),
        total_items: cart.totalQuantity,
        total_price_cents: toCents(cart.cost.totalAmount.amount),
        currency: cart.cost.totalAmount.currencyCode
      } : {
        id: null,
        items: [],
        total_items: 0,
        total_price_cents: 0,
        currency: 'USD'
      },
      // Include completed order if checkout was completed
      last_order: completedOrder || null,
      timestamp: new Date().toISOString()
    };

    return NextResponse.json(state, {
      status: 200,
      headers: { 'Cache-Control': 'no-store' }
    });
  } catch (error) {
    console.error('State endpoint error:', error);
    return NextResponse.json({
      cart: { id: null, items: [], total_items: 0, total_price_cents: 0, currency: 'USD' },
      last_order: null,
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    }, {
      status: 500,
      headers: { 'Cache-Control': 'no-store' }
    });
  }
}
