import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

export async function POST(request: NextRequest) {
  const refreshToken = request.cookies.get('cc_refresh_token')?.value;

  if (!refreshToken) {
    return NextResponse.json({ error: { code: 'NO_SESSION', message: 'No active session' } }, { status: 401 });
  }

  const backendRes = await fetch(`${BACKEND_URL}/api/v1/auth/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });

  const data = await backendRes.json();

  if (!backendRes.ok) {
    // Refresh failed — clear cookies and require re-login
    const response = NextResponse.json(
      { error: { code: 'SESSION_EXPIRED', message: 'Session expired' } },
      { status: 401 }
    );
    response.cookies.delete('cc_access_token');
    response.cookies.delete('cc_refresh_token');
    return response;
  }

  const { access_token, refresh_token, expires_in } = data.data;
  const isProduction = process.env.NODE_ENV === 'production';
  const cookieOptions = {
    httpOnly: true,
    secure: isProduction,
    sameSite: 'lax' as const,
    path: '/',
  };

  const response = NextResponse.json({ success: true });
  response.cookies.set('cc_access_token', access_token, {
    ...cookieOptions,
    maxAge: expires_in ?? 900,
  });
  response.cookies.set('cc_refresh_token', refresh_token, {
    ...cookieOptions,
    maxAge: 7 * 24 * 60 * 60,
  });

  return response;
}
