import { getCart, saveCompletedOrder, setStoredCart } from 'lib/shopify';
import { Cart } from 'lib/shopify/types';
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';

// GET requests should redirect to checkout page (not show blank)
export async function GET() {
  redirect('/checkout');
}

// Robust string-to-cents conversion (avoids float precision issues)
function toCents(amount: string): number {
  const parts = amount.split('.');
  const dollars = parts[0] || '0';
  const cents = parts[1] || '00';
  return parseInt(dollars, 10) * 100 + parseInt(cents.padEnd(2, '0').slice(0, 2), 10);
}

export async function POST(request: Request) {
  // Get form data (name and email are required)
  const formData = await request.formData();
  const name = formData.get('name')?.toString()?.trim();
  const email = formData.get('email')?.toString()?.trim();

  // Validate required fields
  if (!name || !email) {
    // Redirect back to checkout with error
    return redirect('/checkout?error=missing_fields');
  }

  // Simple email validation
  if (!email.includes('@') || !email.includes('.')) {
    return redirect('/checkout?error=invalid_email');
  }

  // Get current cart before clearing
  const cart = await getCart();
  const cookieStore = await cookies();
  const cartId = cookieStore.get('cartId')?.value;

  // Save the completed order with customer info
  if (cart && cartId) {
    const order = {
      id: `order_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
      customer: {
        name,
        email
      },
      items: cart.lines.map(line => {
        const lineTotalCents = toCents(line.cost.totalAmount.amount);
        const unitPriceCents = line.merchandise.price?.amount
          ? toCents(line.merchandise.price.amount)
          : (line.quantity > 0 ? Math.round(lineTotalCents / line.quantity) : 0);
        return {
          id: line.merchandise.id,
          slug: line.merchandise.product.handle,
          title: line.merchandise.product.title,
          variant: line.merchandise.title,
          quantity: line.quantity,
          unit_price_cents: unitPriceCents,
          line_total_cents: lineTotalCents
        };
      }),
      total_items: cart.totalQuantity,
      total_price_cents: toCents(cart.cost.totalAmount.amount),
      currency: cart.cost.totalAmount.currencyCode,
      completed_at: new Date().toISOString()
    };

    // Save using the cartId as the session key
    saveCompletedOrder(cartId, order);
  }

  // Reset cart to empty under same cartId (parallel-safe, keeps cookie)
  if (cartId) {
    const emptyCart: Cart = {
      id: cartId,
      checkoutUrl: '/checkout',
      cost: {
        subtotalAmount: { amount: '0.00', currencyCode: 'USD' },
        totalAmount: { amount: '0.00', currencyCode: 'USD' },
        totalTaxAmount: { amount: '0.00', currencyCode: 'USD' }
      },
      lines: [],
      totalQuantity: 0
    };
    setStoredCart(cartId, emptyCart);
  }

  // Redirect to confirmation page
  redirect('/checkout/confirmation');
}
