import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const userAgent = request.headers.get('user-agent') || '';
  const url = request.nextUrl;

  // Allow ?mode=human to bypass agent routing
  if (url.searchParams.get('mode') === 'human') {
    return NextResponse.next();
  }

  // Detect bot/agent user agents
  const isBot = /GPTBot|BenchmarkAgent|Claude|Anthropic|OpenAI/i.test(userAgent);

  // Rewrite root to /agent for detected bots
  if (isBot && url.pathname === '/') {
    return NextResponse.rewrite(new URL('/agent', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: '/'
};
