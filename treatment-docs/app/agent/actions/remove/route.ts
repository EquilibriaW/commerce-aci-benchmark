import { removeFromCart, getCart } from 'lib/shopify';
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
    let lineId: string | null = null;
    let isFormSubmit = false;

    if (contentType.includes('application/json')) {
      const body = await request.json();
      lineId = body.lineId;
    } else {
      // Form submission
      isFormSubmit = true;
      const formData = await request.formData();
      lineId = formData.get('lineId') as string;
    }

    if (!lineId) {
      if (isFormSubmit) {
        redirect('/agent?error=missing_lineId');
      }
      return NextResponse.json({
        success: false,
        error: 'Missing lineId parameter'
      }, { status: 400 });
    }

    // Remove from cart
    await removeFromCart([lineId]);

    // For form submissions, redirect back to agent page
    if (isFormSubmit) {
      redirect('/agent');
    }

    // For JSON API calls, return cart state
    const cart = await getCart();

    return NextResponse.json({
      success: true,
      message: 'Item removed from cart',
      cart: cart ? {
        id: cart.id,
        total_items: cart.totalQuantity,
        total_price: parseFloat(cart.cost.totalAmount.amount),
        currency: cart.cost.totalAmount.currencyCode,
        items: cart.lines.map(line => ({
          id: line.merchandise.id,
          lineId: line.id,
          slug: line.merchandise.product.handle,
          title: line.merchandise.product.title,
          variant: line.merchandise.title,
          quantity: line.quantity
        }))
      } : null,
      timestamp: new Date().toISOString()
    }, {
      status: 200,
      headers: { 'Cache-Control': 'no-store' }
    });
  } catch (error) {
    if (error instanceof Error && error.message === 'NEXT_REDIRECT') {
      throw error;
    }

    console.error('Remove from cart error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to remove item',
      timestamp: new Date().toISOString()
    }, {
      status: 500,
      headers: { 'Cache-Control': 'no-store' }
    });
  }
}
