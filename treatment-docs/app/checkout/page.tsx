import { getCart } from 'lib/shopify';
import { redirect } from 'next/navigation';
import Image from 'next/image';
import Price from 'components/price';

export const metadata = {
  title: 'Checkout',
  description: 'Complete your order'
};

export default async function CheckoutPage({
  searchParams
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  const cart = await getCart();
  const { error } = await searchParams;

  if (!cart || cart.lines.length === 0) {
    redirect('/');
  }

  const errorMessage = error === 'missing_fields'
    ? 'Please fill in both name and email fields.'
    : error === 'invalid_email'
      ? 'Please enter a valid email address.'
      : null;

  return (
    <div className="mx-auto max-w-2xl px-4 py-16 sm:px-6 lg:px-8">
      <h1 className="text-3xl font-bold tracking-tight text-black dark:text-white">
        Checkout
      </h1>

      <div className="mt-12">
        <h2 className="text-lg font-medium text-black dark:text-white">Order Summary</h2>

        <ul className="mt-6 divide-y divide-neutral-200 border-t border-b border-neutral-200 dark:divide-neutral-700 dark:border-neutral-700">
          {cart.lines.map((item) => (
            <li key={item.id} className="flex py-6">
              <div className="relative h-24 w-24 flex-shrink-0 overflow-hidden rounded-md border border-neutral-200 dark:border-neutral-700">
                {item.merchandise.product.featuredImage && (
                  <Image
                    src={item.merchandise.product.featuredImage.url}
                    alt={item.merchandise.product.featuredImage.altText || item.merchandise.product.title}
                    fill
                    className="object-cover object-center"
                  />
                )}
              </div>
              <div className="ml-4 flex flex-1 flex-col">
                <div className="flex justify-between text-base font-medium text-black dark:text-white">
                  <h3>{item.merchandise.product.title}</h3>
                  <Price
                    amount={item.cost.totalAmount.amount}
                    currencyCode={item.cost.totalAmount.currencyCode}
                  />
                </div>
                <p className="mt-1 text-sm text-neutral-500">
                  {item.merchandise.title}
                </p>
                <p className="mt-1 text-sm text-neutral-500">
                  Qty: {item.quantity}
                </p>
              </div>
            </li>
          ))}
        </ul>

        <div className="mt-6 space-y-4">
          <div className="flex justify-between text-base font-medium text-black dark:text-white">
            <p>Subtotal</p>
            <Price
              amount={cart.cost.subtotalAmount.amount}
              currencyCode={cart.cost.subtotalAmount.currencyCode}
            />
          </div>
          <div className="flex justify-between text-sm text-neutral-500">
            <p>Shipping</p>
            <p>Calculated at next step</p>
          </div>
          <div className="flex justify-between text-sm text-neutral-500">
            <p>Taxes</p>
            <Price
              amount={cart.cost.totalTaxAmount.amount}
              currencyCode={cart.cost.totalTaxAmount.currencyCode}
            />
          </div>
          <div className="flex justify-between border-t border-neutral-200 pt-4 text-lg font-bold text-black dark:border-neutral-700 dark:text-white">
            <p>Total</p>
            <Price
              amount={cart.cost.totalAmount.amount}
              currencyCode={cart.cost.totalAmount.currencyCode}
            />
          </div>
        </div>

        <div className="mt-10 rounded-lg border border-neutral-200 bg-neutral-50 p-6 dark:border-neutral-700 dark:bg-neutral-900">
          <h3 className="text-lg font-medium text-black dark:text-white">
            Demo Checkout
          </h3>
          <p className="mt-2 text-sm text-neutral-500">
            This is a demo store for AI agent benchmarking. No actual transactions will be processed.
          </p>

          {errorMessage && (
            <div className="mt-4 rounded-md bg-red-50 p-4 dark:bg-red-900/20">
              <p className="text-sm text-red-700 dark:text-red-400">
                {errorMessage}
              </p>
            </div>
          )}

          <form action="/checkout/complete" method="POST" className="mt-6">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">
                  Email <span className="text-red-500">*</span>
                </label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  required
                  placeholder="you@example.com"
                  className="mt-1 block w-full rounded-md border border-neutral-300 px-3 py-2 text-sm placeholder-neutral-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-neutral-600 dark:bg-neutral-800 dark:text-white"
                />
              </div>
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-neutral-700 dark:text-neutral-300">
                  Full Name <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  required
                  placeholder="John Doe"
                  className="mt-1 block w-full rounded-md border border-neutral-300 px-3 py-2 text-sm placeholder-neutral-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-neutral-600 dark:bg-neutral-800 dark:text-white"
                />
              </div>
            </div>
            <button
              type="submit"
              className="mt-6 w-full rounded-full bg-blue-600 px-6 py-3 text-center text-sm font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              Complete Demo Order
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
