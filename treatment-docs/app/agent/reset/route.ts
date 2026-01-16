import { randomUUID } from 'crypto';
import { clearEvents, deleteCompletedOrder, deleteStoredCart, setStoredCart } from 'lib/mock/storage';
import { parseVariantLevel, parseVariantSeed } from 'lib/benchmark/variants';
import { NextRequest, NextResponse } from 'next/server';
import { Cart } from 'lib/shopify/types';

const BENCHMARK_SECRET = process.env.BENCHMARK_SECRET;

export async function POST(request: NextRequest) {
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
    // Delete ONLY the old session (if exists) - NOT all carts/orders
    // This allows parallel benchmark runs without interference
    const oldCartId = request.cookies.get('cartId')?.value;
    if (oldCartId) {
      deleteStoredCart(oldCartId);
      deleteCompletedOrder(oldCartId);
      clearEvents(oldCartId);
    }

    // Pattern B: Server mints fresh session ID
    const newSessionId = randomUUID();

    // CRITICAL: Create empty cart in storage with this ID
    // This ensures addToCart can find and update it (cookies().set is read-only in server actions)
    const emptyCart: Cart = {
      id: newSessionId,
      checkoutUrl: '/checkout',
      cost: {
        subtotalAmount: { amount: '0.00', currencyCode: 'USD' },
        totalAmount: { amount: '0.00', currencyCode: 'USD' },
        totalTaxAmount: { amount: '0.00', currencyCode: 'USD' }
      },
      lines: [],
      totalQuantity: 0
    };
    setStoredCart(newSessionId, emptyCart);

    const variantSeed = parseVariantSeed(request.headers.get('X-Benchmark-Variant-Seed'));
    const variantLevel = parseVariantLevel(request.headers.get('X-Benchmark-Variant-Level'));

    const response = NextResponse.json({
      status: 'reset_complete',
      session_id: newSessionId,
      variant_seed: variantSeed,
      variant_level: variantLevel,
      timestamp: new Date().toISOString()
    }, {
      status: 200,
      headers: { 'Cache-Control': 'no-store' }
    });

    // Set fresh cart cookie
    response.cookies.set('cartId', newSessionId, {
      path: '/',
      httpOnly: true,
      sameSite: 'lax'
    });

    response.cookies.set('variantSeed', String(variantSeed), {
      path: '/',
      httpOnly: true,
      sameSite: 'lax'
    });

    response.cookies.set('variantLevel', String(variantLevel), {
      path: '/',
      httpOnly: true,
      sameSite: 'lax'
    });

    // Set experimental factor cookies from benchmark headers
    const discoverability = request.headers.get('X-Benchmark-Discoverability') || 'navbar';
    const capability = request.headers.get('X-Benchmark-Capability') || 'advantage';

    response.cookies.set('bench_discoverability', discoverability, {
      path: '/',
      httpOnly: true,
      sameSite: 'lax'
    });

    response.cookies.set('bench_capability', capability, {
      path: '/',
      httpOnly: true,
      sameSite: 'lax'
    });

    return response;
  } catch (error) {
    console.error('Reset endpoint error:', error);
    return NextResponse.json({
      status: 'error',
      message: error instanceof Error ? error.message : 'Failed to reset state',
      timestamp: new Date().toISOString()
    }, {
      status: 500,
      headers: { 'Cache-Control': 'no-store' }
    });
  }
}
