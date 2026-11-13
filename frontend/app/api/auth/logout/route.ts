import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

export async function POST(request: NextRequest) {
  const accessToken = request.cookies.get('cc_access_token')?.value;

  // Notify backend to revoke the session
  if (accessToken) {
    await fetch(`${BACKEND_URL}/api/v1/auth/logout`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${accessToken}`,
      },
    }).catch(() => {/* Backend might be unavailable; still clear cookies */});
  }

  const response = NextResponse.json({ success: true });
  response.cookies.delete('cc_access_token');
  response.cookies.delete('cc_refresh_token');

  return response;
}
