import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

export async function POST(request: NextRequest) {
  const body = await request.json();

  // Forward login request to FastAPI
  const backendRes = await fetch(`${BACKEND_URL}/api/v1/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  const data = await backendRes.json();

  if (!backendRes.ok) {
    return NextResponse.json(
      { error: data?.error ?? { code: 'AUTH_ERROR', message: 'Authentication failed' } },
      { status: backendRes.status }
    );
  }

  const { access_token, refresh_token, expires_in } = data.data;

  const response = NextResponse.json({ success: true, message: 'Authenticated' });

  // Set HttpOnly cookies — tokens never exposed to JS
  const isProduction = process.env.NODE_ENV === 'production';
  const cookieOptions = {
    httpOnly: true,
    secure: isProduction,
    sameSite: 'lax' as const,
    path: '/',
  };

  response.cookies.set('cc_access_token', access_token, {
    ...cookieOptions,
    maxAge: expires_in ?? 900,
  });

  response.cookies.set('cc_refresh_token', refresh_token, {
    ...cookieOptions,
    maxAge: 7 * 24 * 60 * 60, // 7 days
  });

  return response;
}
