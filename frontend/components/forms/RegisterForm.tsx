'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useState } from 'react';
import { registerUser } from '@/lib/auth/client-auth';
import { Button } from '@/components/ui/button';

const registerSchema = z
  .object({
    full_name: z.string().min(2, 'Full name must be at least 2 characters'),
    email: z.string().email('Please enter a valid email address'),
    password: z
      .string()
      .min(8, 'Password must be at least 8 characters')
      .regex(/[A-Z]/, 'Must contain at least one uppercase letter')
      .regex(/[0-9]/, 'Must contain at least one number'),
    confirmPassword: z.string(),
  })
  .refine((d) => d.password === d.confirmPassword, {
    message: 'Passwords do not match',
    path: ['confirmPassword'],
  });

type RegisterFormData = z.infer<typeof registerSchema>;

export function RegisterForm() {
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<RegisterFormData>({ resolver: zodResolver(registerSchema) });

  const onSubmit = async (data: RegisterFormData) => {
    setError(null);
    try {
      await registerUser({ email: data.email, password: data.password, full_name: data.full_name });
      setSuccess(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed');
    }
  };

  if (success) {
    return (
      <div className="rounded-lg bg-emerald-50 p-6 text-center space-y-3 border border-emerald-200">
        <h2 className="text-lg font-semibold text-emerald-800">Account created!</h2>
        <p className="text-emerald-700 text-sm">
          A verification link has been sent to your email. Please verify your account before logging in.
        </p>
        <div className="pt-2">
          <a href="/login" className="text-blue-600 font-medium hover:underline text-sm">
            Proceed to sign in →
          </a>
        </div>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="mt-8 space-y-6" noValidate>
      {error && (
        <div role="alert" className="rounded-md bg-rose-50 p-4 text-sm text-rose-700 border border-rose-200">
          {error}
        </div>
      )}
      <div className="space-y-4">
        <div>
          <label htmlFor="full_name" className="block text-sm font-medium text-gray-700">
            Full name
          </label>
          <input
            id="full_name"
            type="text"
            autoComplete="name"
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
            {...register('full_name')}
          />
          {errors.full_name && (
            <p role="alert" className="mt-1 text-xs text-rose-600">
              {errors.full_name.message}
            </p>
          )}
        </div>
        <div>
          <label htmlFor="email" className="block text-sm font-medium text-gray-700">
            Email address
          </label>
          <input
            id="email"
            type="email"
            autoComplete="email"
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
            {...register('email')}
          />
          {errors.email && (
            <p role="alert" className="mt-1 text-xs text-rose-600">
              {errors.email.message}
            </p>
          )}
        </div>
        <div>
          <label htmlFor="password" className="block text-sm font-medium text-gray-700">
            Password (min 8 characters, 1 uppercase, 1 number)
          </label>
          <input
            id="password"
            type="password"
            autoComplete="new-password"
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
            {...register('password')}
          />
          {errors.password && (
            <p role="alert" className="mt-1 text-xs text-rose-600">
              {errors.password.message}
            </p>
          )}
        </div>
        <div>
          <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
            Confirm password
          </label>
          <input
            id="confirmPassword"
            type="password"
            autoComplete="new-password"
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
            {...register('confirmPassword')}
          />
          {errors.confirmPassword && (
            <p role="alert" className="mt-1 text-xs text-rose-600">
              {errors.confirmPassword.message}
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
        {isSubmitting ? 'Creating account…' : 'Create account'}
      </Button>
      <p className="text-center text-sm text-gray-600">
        Already have an account?{' '}
        <a href="/login" className="text-blue-600 hover:underline">
          Sign in
        </a>
      </p>
    </form>
  );
}

