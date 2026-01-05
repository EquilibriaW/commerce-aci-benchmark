import Link from 'next/link';

export const metadata = {
  title: 'Order Confirmed',
  description: 'Your demo order has been confirmed'
};

export default function ConfirmationPage() {
  const orderNumber = `DEMO-${Date.now().toString(36).toUpperCase()}`;

  return (
    <div className="mx-auto max-w-2xl px-4 py-16 sm:px-6 lg:px-8">
      <div className="text-center">
        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-green-100 dark:bg-green-900">
          <svg
            className="h-8 w-8 text-green-600 dark:text-green-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M5 13l4 4L19 7"
            />
          </svg>
        </div>

        <h1 className="mt-6 text-3xl font-bold tracking-tight text-black dark:text-white">
          Order Confirmed!
        </h1>

        <p className="mt-2 text-lg text-neutral-600 dark:text-neutral-400">
          Thank you for your demo order.
        </p>

        <div className="mt-8 rounded-lg border border-neutral-200 bg-neutral-50 p-6 dark:border-neutral-700 dark:bg-neutral-900">
          <p className="text-sm text-neutral-500">Order number</p>
          <p className="mt-1 text-xl font-mono font-bold text-black dark:text-white">
            {orderNumber}
          </p>
        </div>

        <div className="mt-8 rounded-lg border border-yellow-200 bg-yellow-50 p-4 dark:border-yellow-800 dark:bg-yellow-900/20">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            This is a demo store for AI agent benchmarking. No actual order was placed and no payment was processed.
          </p>
        </div>

        <div className="mt-10">
          <Link
            href="/"
            className="inline-block rounded-full bg-blue-600 px-8 py-3 text-sm font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Continue Shopping
          </Link>
        </div>
      </div>
    </div>
  );
}
