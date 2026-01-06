import { updateCart, getCart } from 'lib/shopify';
import { redirect } from 'next/navigation';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Support both form data and JSON body
    const contentType = request.headers.get('content-type') || '';
    let lineId: string | null = null;
    let merchandiseId: string | null = null;
    let quantity: number = 1;
    let isFormSubmit = false;

    if (contentType.includes('application/json')) {
      const body = await request.json();
      lineId = body.lineId;
      merchandiseId = body.merchandiseId;
      quantity = body.quantity ?? 1;
    } else {
      // Form submission
      isFormSubmit = true;
      const formData = await request.formData();
      lineId = formData.get('lineId') as string;
      merchandiseId = formData.get('merchandiseId') as string;
      quantity = parseInt(formData.get('quantity') as string, 10) || 1;
    }

    if (!lineId || !merchandiseId) {
      if (isFormSubmit) {
        redirect('/agent?error=missing_params');
      }
      return NextResponse.json({
        success: false,
        error: 'Missing lineId or merchandiseId parameter'
      }, { status: 400 });
    }

    // Update cart line
    await updateCart([{ id: lineId, merchandiseId, quantity }]);

    // For form submissions, redirect back to agent page
    if (isFormSubmit) {
      redirect('/agent');
    }

    // For JSON API calls, return cart state
    const cart = await getCart();

    return NextResponse.json({
      success: true,
      message: `Updated quantity to ${quantity}`,
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

    console.error('Update cart error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to update cart',
      timestamp: new Date().toISOString()
    }, {
      status: 500,
      headers: { 'Cache-Control': 'no-store' }
    });
  }
}
