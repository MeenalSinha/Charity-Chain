'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { loginUser } from '@/lib/auth/client-auth';
import { Button } from '@/components/ui/button';

const loginSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  password: z.string().min(1, 'Password is required'),
});

type LoginFormData = z.infer<typeof loginSchema>;

export function LoginForm() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<LoginFormData>({ resolver: zodResolver(loginSchema) });

  const onSubmit = async (data: LoginFormData) => {
    setError(null);
    try {
      await loginUser(data);
      router.push('/dashboard/donor');
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="mt-8 space-y-6" noValidate>
      {error && (
        <div role="alert" className="rounded-md bg-rose-50 p-4 text-sm text-rose-700 border border-rose-200">
          {error}
        </div>
      )}
      <div className="space-y-4">
        <div>
          <label htmlFor="email" className="block text-sm font-medium text-gray-700">
            Email address
          </label>
          <input
            id="email"
            type="email"
            autoComplete="email"
            aria-describedby={errors.email ? 'email-error' : undefined}
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
            {...register('email')}
          />
          {errors.email && (
            <p id="email-error" role="alert" className="mt-1 text-xs text-rose-600">
              {errors.email.message}
            </p>
          )}
        </div>
        <div>
          <label htmlFor="password" className="block text-sm font-medium text-gray-700">
            Password
          </label>
          <input
            id="password"
            type="password"
            autoComplete="current-password"
            aria-describedby={errors.password ? 'password-error' : undefined}
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
            {...register('password')}
          />
          {errors.password && (
            <p id="password-error" role="alert" className="mt-1 text-xs text-rose-600">
              {errors.password.message}
            </p>
          )}
        </div>
      </div>
      <Button
        type="submit"
        disabled={isSubmitting}
        className="w-full"
        aria-busy={isSubmitting}
      >
        {isSubmitting ? 'Signing in…' : 'Sign in'}
      </Button>
      <div className="flex items-center justify-between text-sm text-gray-600">
        <a href="/forgot-password" className="text-blue-600 hover:underline">
          Forgot password?
        </a>
        <a href="/register" className="text-blue-600 hover:underline">
          Create an account
        </a>
      </div>
    </form>
  );
}

