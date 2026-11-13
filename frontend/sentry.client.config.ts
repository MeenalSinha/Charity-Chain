/**
 * sentry.client.config.ts - Frontend Sentry Client Configuration
 * ==============================================================
 * Strict zero-PII error monitoring for Next.js browser execution.
 */

export const SENTRY_CONFIG = {
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN || '',
  environment: process.env.NODE_ENV || 'development',
  tracesSampleRate: 0.1,
  replaysSessionSampleRate: 0.0,
  replaysOnErrorSampleRate: 0.1,
  beforeSend(event: any) {
    // Redact tokens, cookies, passwords, and sensitive fields
    if (event.request && event.request.headers) {
      delete event.request.headers['authorization'];
      delete event.request.headers['cookie'];
    }
    if (event.user) {
      delete event.user.email;
      delete event.user.ip_address;
    }
    return event;
  },
};

