import { randomUUID } from 'crypto';
import { cookies } from 'next/headers';

import { appendEvent, SessionEvent } from './storage';

export function getSessionIdFromCookies(): string | undefined {
  return cookies().get('cartId')?.value;
}

export function logEvent(type: string, payload?: Record<string, unknown>): void {
  try {
    const sessionId = getSessionIdFromCookies();
    if (!sessionId) {
      return;
    }

    const event: SessionEvent = {
      id: randomUUID(),
      type,
      at: new Date().toISOString(),
      ...(payload ? { payload } : {})
    };

    appendEvent(sessionId, event);
  } catch (error) {
    console.error('Failed to log event:', error);
  }
}
