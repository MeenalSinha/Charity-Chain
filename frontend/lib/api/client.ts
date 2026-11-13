import { generateCorrelationId } from '@/lib/utilities';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000/api/v1';
const REQUEST_TIMEOUT_MS = 30_000;
const MAX_RETRIES = 2;
const RETRYABLE_STATUS = new Set([429, 503]);

export class ApiClientError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly statusCode: number,
    public readonly correlationId?: string,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = 'ApiClientError';
  }
}

export interface RequestOptions {
  token?: string;
  signal?: AbortSignal;
  timeout?: number;
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit & { timeout?: number }
): Promise<Response> {
  const { timeout = REQUEST_TIMEOUT_MS, ...fetchOptions } = options;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    return await fetch(url, {
      ...fetchOptions,
      signal: options.signal ?? controller.signal,
    });
  } finally {
    clearTimeout(id);
  }
}

function buildHeaders(token?: string, correlationId?: string): HeadersInit {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-Correlation-ID': correlationId ?? generateCorrelationId(),
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

async function parseResponse<T>(response: Response): Promise<T> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new ApiClientError(
      'PARSE_ERROR',
      'Failed to parse server response',
      response.status
    );
  }

  if (!response.ok) {
    const err = body as { error?: { code?: string; message?: string; details?: unknown }; correlation_id?: string; detail?: string };
    const code = err?.error?.code ?? 'API_ERROR';
    const message = err?.error?.message ?? err?.detail ?? `HTTP ${response.status}`;
    throw new ApiClientError(
      code,
      message,
      response.status,
      err?.correlation_id,
      err?.error?.details
    );
  }

  const wrapper = body as { success?: boolean; data?: T };
  if (wrapper?.data !== undefined) {
    return wrapper.data;
  }
  return body as T;
}

async function request<T>(
  method: string,
  path: string,
  options: RequestOptions & { body?: unknown } = {}
): Promise<T> {
  const { token, body, signal, timeout } = options;
  const correlationId = generateCorrelationId();
  const url = `${API_BASE_URL}${path}`;

  let lastError: Error | null = null;
  const isIdempotent = method === 'GET' || method === 'HEAD';

  for (let attempt = 0; attempt <= (isIdempotent ? MAX_RETRIES : 0); attempt++) {
    if (attempt > 0) {
      await new Promise((r) => setTimeout(r, 1000 * attempt));
    }
    try {
      const response = await fetchWithTimeout(url, {
        method,
        headers: buildHeaders(token, correlationId),
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal,
        timeout,
      });

      if (attempt < MAX_RETRIES && isIdempotent && RETRYABLE_STATUS.has(response.status)) {
        lastError = new ApiClientError('RETRYABLE', `HTTP ${response.status}`, response.status, correlationId);
        continue;
      }

      return await parseResponse<T>(response);
    } catch (err) {
      if (err instanceof ApiClientError) {
        lastError = err;
        if (isIdempotent && RETRYABLE_STATUS.has(err.statusCode)) continue;
        throw err;
      }
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw new ApiClientError('TIMEOUT', 'Request timed out or was cancelled', 408, correlationId);
      }
      throw err;
    }
  }
  throw lastError ?? new ApiClientError('UNKNOWN', 'Unknown error', 500, correlationId);
}

export async function uploadFile<T>(
  path: string,
  formData: FormData,
  options: RequestOptions = {}
): Promise<T> {
  const { token, signal, timeout } = options;
  const correlationId = generateCorrelationId();
  const url = `${API_BASE_URL}${path}`;

  const headers: Record<string, string> = {
    'X-Correlation-ID': correlationId,
  };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const response = await fetchWithTimeout(url, {
    method: 'POST',
    headers,
    body: formData,
    signal,
    timeout: timeout ?? 60_000,
  });

  return parseResponse<T>(response);
}

export const apiClient = {
  get: <T>(path: string, options?: RequestOptions) => request<T>('GET', path, options),
  post: <T>(path: string, body?: unknown, options?: RequestOptions) => request<T>('POST', path, { body, ...options }),
  put: <T>(path: string, body?: unknown, options?: RequestOptions) => request<T>('PUT', path, { body, ...options }),
  patch: <T>(path: string, body?: unknown, options?: RequestOptions) => request<T>('PATCH', path, { body, ...options }),
  delete: <T>(path: string, options?: RequestOptions) => request<T>('DELETE', path, options),
  upload: uploadFile,
};

