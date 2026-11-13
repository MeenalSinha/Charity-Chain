import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const accessToken = request.cookies.get('cc_access_token')?.value;

  if (!accessToken) {
    // Try to refresh
    const refreshToken = request.cookies.get('cc_refresh_token')?.value;
    if (!refreshToken) {
      return NextResponse.json({ authenticated: false }, { status: 401 });
    }
    // Client should call /api/auth/refresh instead
    return NextResponse.json({ authenticated: false, needs_refresh: true }, { status: 401 });
  }

  // Return token to client for making authenticated requests
  // This is acceptable because: same-origin, lax SameSite, and the token is short-lived
  return NextResponse.json({ authenticated: true, access_token: accessToken });
}
