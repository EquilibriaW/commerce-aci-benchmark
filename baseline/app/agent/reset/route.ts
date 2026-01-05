export const runtime = 'nodejs'; // CRITICAL: Ensures consistent server instance

import { randomUUID } from 'crypto';
import { deleteStoredCart } from 'lib/mock/storage';
import { NextRequest, NextResponse } from 'next/server';

const BENCHMARK_SECRET = process.env.BENCHMARK_SECRET;

export async function POST(request: NextRequest) {
  // Fail closed - require secret
  if (!BENCHMARK_SECRET) {
    return NextResponse.json(
      { error: 'Server misconfigured', message: 'BENCHMARK_SECRET not set' },
      { status: 500, headers: { 'Cache-Control': 'no-store' } }
    );
  }

  // Security check - require benchmark secret header
  const providedSecret = request.headers.get('X-Benchmark-Secret');
  if (providedSecret !== BENCHMARK_SECRET) {
    return NextResponse.json(
      { error: 'Forbidden', message: 'Invalid or missing X-Benchmark-Secret header' },
      { status: 403, headers: { 'Cache-Control': 'no-store' } }
    );
  }

  try {
    // Delete ONLY the old session (if exists) - NOT all carts
    // This allows parallel benchmark runs without interference
    const oldCartId = request.cookies.get('cartId')?.value;
    if (oldCartId) {
      deleteStoredCart(oldCartId);
    }

    // Pattern B: Server mints fresh session ID
    const newSessionId = randomUUID();

    const response = NextResponse.json({
      status: 'reset_complete',
      session_id: newSessionId,
      timestamp: new Date().toISOString()
    }, {
      status: 200,
      headers: { 'Cache-Control': 'no-store' }
    });

    // Set fresh cart cookie
    response.cookies.set('cartId', newSessionId, {
      path: '/',
      httpOnly: true,
      sameSite: 'lax'
    });

    return response;
  } catch (error) {
    console.error('Reset endpoint error:', error);
    return NextResponse.json({
      status: 'error',
      message: error instanceof Error ? error.message : 'Failed to reset state',
      timestamp: new Date().toISOString()
    }, {
      status: 500,
      headers: { 'Cache-Control': 'no-store' }
    });
  }
}
