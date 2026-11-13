import { describe, it, expect } from 'vitest';
import { ApiClientError } from '@/lib/api/client';

describe('ApiClientError', () => {
  it('correctly constructs with code, message, and statusCode', () => {
    const err = new ApiClientError('AUTH_REQUIRED', 'Please sign in', 401, 'corr-999');
    expect(err.code).toBe('AUTH_REQUIRED');
    expect(err.message).toBe('Please sign in');
    expect(err.statusCode).toBe(401);
    expect(err.correlationId).toBe('corr-999');
    expect(err.name).toBe('ApiClientError');
  });
});

