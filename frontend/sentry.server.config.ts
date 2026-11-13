/**
 * sentry.server.config.ts - Next.js Server-Side Sentry Configuration
 * ==================================================================
 * Captures uncaught route handler exceptions with strict credential scrubbing.
 */

export const SENTRY_SERVER_CONFIG = {
  dsn: process.env.SENTRY_DSN || '',
  environment: process.env.NODE_ENV || 'development',
  tracesSampleRate: 0.1,
  beforeSend(event: any) {
    if (event.request && event.request.headers) {
      delete event.request.headers['cookie'];
      delete event.request.headers['authorization'];
      delete event.request.headers['x-api-key'];
    }
    return event;
  },
};

