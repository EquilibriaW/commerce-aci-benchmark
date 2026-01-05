import { addToCart, createCart, getCart } from 'lib/shopify';
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Support both form data and JSON body
    const contentType = request.headers.get('content-type') || '';
    let variantId: string | null = null;
    let quantity: number = 1;
    let isFormSubmit = false;

    if (contentType.includes('application/json')) {
      const body = await request.json();
      variantId = body.variantId;
      quantity = body.quantity || 1;
    } else {
      // Form submission
      isFormSubmit = true;
      const formData = await request.formData();
      variantId = formData.get('variantId') as string;
      quantity = parseInt(formData.get('quantity') as string, 10) || 1;
    }

    if (!variantId) {
      if (isFormSubmit) {
        redirect('/agent?error=missing_variant');
      }
      return NextResponse.json({
        success: false,
        error: 'Missing variantId parameter'
      }, { status: 400 });
    }

    // Ensure cart exists
    const cookieStore = await cookies();
    let cartId = cookieStore.get('cartId')?.value;

    if (!cartId) {
      const cart = await createCart();
      cartId = cart.id!;
      cookieStore.set('cartId', cartId);
    }

    // Add to cart
    await addToCart([{ merchandiseId: variantId, quantity }]);

    // For form submissions, redirect back to agent page
    if (isFormSubmit) {
      redirect('/agent');
    }

    // For JSON API calls, return cart state
    const cart = await getCart();

    return NextResponse.json({
      success: true,
      message: `Added ${quantity} item(s) to cart`,
      cart: cart ? {
        id: cart.id,
        total_items: cart.totalQuantity,
        total_price: parseFloat(cart.cost.totalAmount.amount),
        currency: cart.cost.totalAmount.currencyCode,
        items: cart.lines.map(line => ({
          id: line.merchandise.id,
          slug: line.merchandise.product.handle,
          title: line.merchandise.product.title,
          variant: line.merchandise.title,
          quantity: line.quantity
        }))
      } : null,
      timestamp: new Date().toISOString()
    }, {
      status: 200,
      headers: {
        'Cache-Control': 'no-store'
      }
    });
  } catch (error) {
    // Check if it's a redirect (Next.js throws for redirects)
    if (error instanceof Error && error.message === 'NEXT_REDIRECT') {
      throw error;
    }

    console.error('Add to cart error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to add item to cart',
      timestamp: new Date().toISOString()
    }, {
      status: 500,
      headers: {
        'Cache-Control': 'no-store'
      }
    });
  }
}
