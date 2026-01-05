import { clearAllCarts } from 'lib/shopify';
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';

export async function POST() {
  // Clear the cart after "checkout"
  const cookieStore = await cookies();
  cookieStore.delete('cartId');
  clearAllCarts();

  // Redirect to confirmation page
  redirect('/checkout/confirmation');
}
