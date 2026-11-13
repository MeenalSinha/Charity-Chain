'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { useState, Suspense } from 'react';
import Link from 'next/link';
import { resetPassword } from '@/lib/auth/client-auth';
import { Button } from '@/components/ui/button';

function ResetPasswordForm() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const token = searchParams.get('token') ?? '';

  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) {
      setError('Invalid or missing password reset token.');
      return;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters long.');
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      await resetPassword(token, password);
      setSuccess(true);
      setTimeout(() => router.push('/login'), 2500);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Password reset failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10 border border-gray-100">
      <div className="text-center mb-6">
        <Link href="/" className="inline-block">
          <span className="text-2xl font-bold text-blue-600">CharityChain</span>
        </Link>
        <h1 className="mt-2 text-xl font-semibold text-gray-900">Set new password</h1>
      </div>

      {success ? (
        <div className="rounded-md bg-emerald-50 p-4 border border-emerald-200 text-center space-y-2">
          <p className="text-sm font-semibold text-emerald-800">Password successfully updated!</p>
          <p className="text-xs text-emerald-600">Redirecting to login…</p>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div role="alert" className="rounded-md bg-rose-50 p-3 text-sm text-rose-700 border border-rose-200">
              {error}
            </div>
          )}
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700">
              New Password
            </label>
            <input
              id="password"
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
            />
          </div>
          <div>
            <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
              Confirm New Password
            </label>
            <input
              id="confirmPassword"
              type="password"
              required
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
            />
          </div>
          <Button type="submit" disabled={loading} className="w-full">
            {loading ? 'Updating password…' : 'Update password'}
          </Button>
          <div className="text-center">
            <Link href="/login" className="text-sm text-blue-600 hover:underline">
              Back to Sign In
            </Link>
          </div>
        </form>
      )}
    </div>
  );
}

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={<div className="text-center p-8 text-gray-500">Loading…</div>}>
      <ResetPasswordForm />
    </Suspense>
  );
}

