import { getCart } from 'lib/shopify';
import { NextRequest, NextResponse } from 'next/server';

const BENCHMARK_SECRET = process.env.BENCHMARK_SECRET || 'sk-bench-123';

export async function GET(request: NextRequest) {
  // Security check - require benchmark secret header
  const providedSecret = request.headers.get('X-Benchmark-Secret');
  if (providedSecret !== BENCHMARK_SECRET) {
    return NextResponse.json(
      { error: 'Forbidden', message: 'Invalid or missing X-Benchmark-Secret header' },
      { status: 403 }
    );
  }

  try {
    const cart = await getCart();

    // Determine page type based on user-agent or query param
    const userAgent = request.headers.get('user-agent') || '';
    const isBot = /GPTBot|BenchmarkAgent|Claude|Anthropic|OpenAI/i.test(userAgent);
    const modeParam = request.nextUrl.searchParams.get('mode');
    const pageType = modeParam === 'human' ? 'human_view' : (isBot ? 'agent_view' : 'human_view');

    const state = {
      cart: cart ? {
        id: cart.id,
        items: cart.lines.map(line => ({
          id: line.merchandise.id,
          slug: line.merchandise.product.handle,
          title: line.merchandise.product.title,
          variant: line.merchandise.title,
          quantity: line.quantity,
          price: parseFloat(line.cost.totalAmount.amount)
        })),
        total_items: cart.totalQuantity,
        total_price: parseFloat(cart.cost.totalAmount.amount),
        currency: cart.cost.totalAmount.currencyCode
      } : {
        id: null,
        items: [],
        total_items: 0,
        total_price: 0,
        currency: 'USD'
      },
      last_action_status: 'success',
      page_type: pageType,
      timestamp: new Date().toISOString()
    };

    return NextResponse.json(state, {
      status: 200,
      headers: {
        'Cache-Control': 'no-store'
      }
    });
  } catch (error) {
    console.error('State endpoint error:', error);
    return NextResponse.json({
      cart: { id: null, items: [], total_items: 0, total_price: 0, currency: 'USD' },
      last_action_status: 'error',
      error: error instanceof Error ? error.message : 'Unknown error',
      page_type: 'agent_view',
      timestamp: new Date().toISOString()
    }, {
      status: 500,
      headers: {
        'Cache-Control': 'no-store'
      }
    });
  }
}
