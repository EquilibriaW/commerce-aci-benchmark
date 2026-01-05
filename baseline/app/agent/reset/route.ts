import { clearAllCarts } from 'lib/shopify';
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
    // Clear all stored carts (file-based storage)
    clearAllCarts();

    // Create response and FORCE cookie deletion on the client
    const response = NextResponse.json({
      status: 'reset_complete',
      message: 'Cart and session state cleared',
      timestamp: new Date().toISOString()
    }, {
      status: 200,
      headers: {
        'Cache-Control': 'no-store'
      }
    });

    // Explicitly expire the cartId cookie on the client side
    response.cookies.set('cartId', '', { maxAge: 0, path: '/' });

    return response;
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
