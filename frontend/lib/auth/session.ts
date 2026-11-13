import { cookies } from 'next/headers';
import { cache } from 'react';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

export interface SessionUser {
  id: string;
  email: string;
  full_name: string;
  role: 'DONOR' | 'REVIEWER' | 'PLATFORM_ADMIN';
  status: string;
}

/**
 * Get the current user from the server session. Returns null if not authenticated.
 * Cached per request using React cache().
 */
export const getCurrentUser = cache(async (): Promise<SessionUser | null> => {
  const cookieStore = await cookies();
  const accessToken = cookieStore.get('cc_access_token')?.value;

  if (!accessToken) return null;

  try {
    const res = await fetch(`${BACKEND_URL}/api/v1/users/me`, {
      headers: {
        Authorization: `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
      next: { revalidate: 0 },
    });

    if (!res.ok) return null;
    const data = await res.json();
    return data?.data ?? null;
  } catch {
    return null;
  }
});

/**
 * Check if the current session has a specific role.
 */
export async function requireRole(
  ...roles: SessionUser['role'][]
): Promise<SessionUser> {
  const user = await getCurrentUser();
  if (!user) {
    throw new Error('UNAUTHENTICATED');
  }
  if (!roles.includes(user.role)) {
    throw new Error('UNAUTHORIZED');
  }
  return user;
}

