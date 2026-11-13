import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

const PROTECTED_PREFIXES = ['/dashboard'];
const AUTH_ROUTES = ['/login', '/register', '/forgot-password', '/reset-password', '/verify-email'];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const token = request.cookies.get('cc_access_token')?.value;
  const isAuthenticated = !!token;

  // If already authenticated and trying to visit login/register, redirect to dashboard
  if (isAuthenticated && AUTH_ROUTES.some((route) => pathname.startsWith(route))) {
    return NextResponse.redirect(new URL('/dashboard/donor', request.url));
  }

  // If not authenticated and visiting protected dashboard route, redirect to login
  if (!isAuthenticated && PROTECTED_PREFIXES.some((prefix) => pathname.startsWith(prefix))) {
    const loginUrl = new URL('/login', request.url);
    loginUrl.searchParams.set('redirect', pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    '/((?!_next/static|_next/image|favicon.ico|api/auth).*)',
  ],
};

