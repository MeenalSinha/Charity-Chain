import { cookies } from 'next/headers';
import { apiClient, ApiClientError } from './client';

/**
 * Get the access token from the server-side session cookie.
 * Returns null if no session is active.
 */
export async function getServerToken(): Promise<string | null> {
  const cookieStore = await cookies();
  return cookieStore.get('cc_access_token')?.value ?? null;
}

/**
 * Authenticated server-side API client.
 * Reads the access token from HttpOnly session cookies automatically.
 */
export const serverApi = {
  get: async <T>(path: string) => {
    const token = await getServerToken();
    return apiClient.get<T>(path, { token: token ?? undefined });
  },
  post: async <T>(path: string, body?: unknown) => {
    const token = await getServerToken();
    return apiClient.post<T>(path, body, { token: token ?? undefined });
  },
  put: async <T>(path: string, body?: unknown) => {
    const token = await getServerToken();
    return apiClient.put<T>(path, body, { token: token ?? undefined });
  },
  patch: async <T>(path: string, body?: unknown) => {
    const token = await getServerToken();
    return apiClient.patch<T>(path, body, { token: token ?? undefined });
  },
  delete: async <T>(path: string) => {
    const token = await getServerToken();
    return apiClient.delete<T>(path, { token: token ?? undefined });
  },
};

export { ApiClientError };

