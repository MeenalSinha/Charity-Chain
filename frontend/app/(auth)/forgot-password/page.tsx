'use client';

import { useState } from 'react';
import Link from 'next/link';
import { forgotPassword } from '@/lib/auth/client-auth';
import { Button } from '@/components/ui/button';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [sent, setSent] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await forgotPassword(email);
      setSent(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send recovery instructions');
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
        <h1 className="mt-2 text-xl font-semibold text-gray-900">Reset your password</h1>
        <p className="mt-1 text-sm text-gray-500">
          Enter your registered email address to receive password reset instructions.
        </p>
      </div>

      {sent ? (
        <div className="rounded-md bg-emerald-50 p-4 border border-emerald-200 text-center space-y-3">
          <p className="text-sm text-emerald-800">
            If an account exists with <strong>{email}</strong>, a password reset link has been dispatched.
          </p>
          <Link href="/login" className="inline-block text-sm text-blue-600 hover:underline">
            Back to Sign In
          </Link>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div role="alert" className="rounded-md bg-rose-50 p-3 text-sm text-rose-700 border border-rose-200">
              {error}
            </div>
          )}
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
              Email address
            </label>
            <input
              id="email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
              placeholder="you@example.com"
            />
          </div>
          <Button type="submit" disabled={loading} className="w-full">
            {loading ? 'Sending instructions…' : 'Send reset link'}
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

