'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { useEffect, useState, Suspense } from 'react';
import Link from 'next/link';

function VerifyEmailContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const token = searchParams.get('token');
  const [status, setStatus] = useState<'verifying' | 'success' | 'error'>('verifying');
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (!token) {
      setStatus('error');
      setMessage('No verification token provided in the link.');
      return;
    }

    fetch('/api/auth/verify-email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    })
      .then(async (res) => {
        if (res.ok) {
          setStatus('success');
          setMessage('Email verified successfully! You can now sign in.');
          setTimeout(() => router.push('/login'), 2500);
        } else {
          const data = await res.json();
          setStatus('error');
          setMessage(data?.error?.message ?? data?.detail ?? 'Verification failed or expired.');
        }
      })
      .catch(() => {
        setStatus('error');
        setMessage('Network error during verification.');
      });
  }, [token, router]);

  return (
    <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10 border border-gray-100 text-center space-y-4">
      <Link href="/" className="inline-block">
        <span className="text-2xl font-bold text-blue-600">CharityChain</span>
      </Link>
      <h1 className="text-xl font-semibold text-gray-900">Email Verification</h1>

      {status === 'verifying' && (
        <div className="py-6 space-y-2">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-600 border-t-transparent" />
          <p className="text-sm text-gray-600">Verifying your token with CharityChain…</p>
        </div>
      )}

      {status === 'success' && (
        <div className="rounded-md bg-emerald-50 p-4 border border-emerald-200 text-emerald-800 text-sm">
          <p className="font-semibold">{message}</p>
          <p className="mt-1 text-xs text-emerald-600">Redirecting to login shortly…</p>
        </div>
      )}

      {status === 'error' && (
        <div className="rounded-md bg-rose-50 p-4 border border-rose-200 text-rose-800 text-sm">
          <p className="font-semibold">Verification Problem</p>
          <p className="mt-1">{message}</p>
          <div className="mt-4">
            <Link href="/login" className="text-blue-600 hover:underline text-xs">
              Go to Sign In
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}

export default function VerifyEmailPage() {
  return (
    <Suspense fallback={<div className="text-center p-8 text-gray-500">Loading verification…</div>}>
      <VerifyEmailContent />
    </Suspense>
  );
}

