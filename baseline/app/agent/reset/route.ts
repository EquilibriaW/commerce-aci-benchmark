import { clearAllCarts } from 'lib/shopify';
import { cookies } from 'next/headers';
import { NextRequest, NextResponse } from 'next/server';

const BENCHMARK_SECRET = process.env.BENCHMARK_SECRET || 'sk-bench-123';

export async function POST(request: NextRequest) {
  // Security check - require benchmark secret header
  const providedSecret = request.headers.get('X-Benchmark-Secret');
  if (providedSecret !== BENCHMARK_SECRET) {
    return NextResponse.json(
      { error: 'Forbidden', message: 'Invalid or missing X-Benchmark-Secret header' },
      { status: 403 }
    );
  }

  try {
    const cookieStore = await cookies();

    // Clear the cart cookie
    cookieStore.delete('cartId');

    // Clear all stored carts (file-based storage)
    clearAllCarts();

    // Clear any other session-related cookies if they exist
    const allCookies = cookieStore.getAll();
    for (const cookie of allCookies) {
      if (cookie.name.startsWith('cart') || cookie.name.startsWith('session')) {
        cookieStore.delete(cookie.name);
      }
    }

    return NextResponse.json({
      status: 'reset_complete',
      message: 'Cart and session state cleared',
      timestamp: new Date().toISOString()
    }, {
      status: 200,
      headers: {
        'Cache-Control': 'no-store'
      }
    });
  } catch (error) {
    console.error('Reset endpoint error:', error);
    return NextResponse.json({
      status: 'error',
      message: error instanceof Error ? error.message : 'Failed to reset state',
      timestamp: new Date().toISOString()
    }, {
      status: 500,
      headers: {
        'Cache-Control': 'no-store'
      }
    });
  }
}
