import { getCart, removeFromCart } from 'lib/shopify';
import { redirect } from 'next/navigation';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  // Gate by capability_mode - block in parity mode
  const capability = request.cookies.get('bench_capability')?.value;
  if (capability === 'parity') {
    const contentType = request.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      return NextResponse.json({
        success: false,
        error: 'capability_parity'
      }, { status: 403, headers: { 'Cache-Control': 'no-store' } });
    } else {
      redirect('/agent?error=capability_parity');
    }
  }

  try {
    // Support both form data and JSON body
    const contentType = request.headers.get('content-type') || '';
    let isFormSubmit = !contentType.includes('application/json');

    // Get current cart and remove all lines
    const cart = await getCart();

    if (cart && cart.lines.length > 0) {
      const lineIds = cart.lines.map(line => line.id!).filter(Boolean);
      if (lineIds.length > 0) {
        await removeFromCart(lineIds);
      }
    }

    // For form submissions, redirect back to agent page
    if (isFormSubmit) {
      redirect('/agent');
    }

    // For JSON API calls, return empty cart state
    return NextResponse.json({
      success: true,
      message: 'Cart cleared',
      cart: {
        id: cart?.id || null,
        total_items: 0,
        total_price: 0,
        currency: 'USD',
        items: []
      },
      timestamp: new Date().toISOString()
    }, {
      status: 200,
      headers: { 'Cache-Control': 'no-store' }
    });
  } catch (error) {
    if (error instanceof Error && error.message === 'NEXT_REDIRECT') {
      throw error;
    }

    console.error('Clear cart error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to clear cart',
      timestamp: new Date().toISOString()
    }, {
      status: 500,
      headers: { 'Cache-Control': 'no-store' }
    });
  }
}
