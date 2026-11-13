import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

export async function POST(request: NextRequest) {
  const body = await request.json();
  const backendRes = await fetch(`${BACKEND_URL}/api/v1/auth/forgot-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await backendRes.json();
  return NextResponse.json(data, { status: backendRes.status });
}

