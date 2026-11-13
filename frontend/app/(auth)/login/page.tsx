import { Metadata } from 'next';
import { LoginForm } from '@/components/forms/LoginForm';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Sign In | CharityChain',
  robots: { index: false },
};

export default function LoginPage() {
  return (
    <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10 border border-gray-100">
      <div className="text-center mb-6">
        <Link href="/" className="inline-block">
          <span className="text-2xl font-bold text-blue-600">CharityChain</span>
        </Link>
        <h1 className="mt-2 text-xl font-semibold text-gray-900">Sign in to your account</h1>
        <p className="mt-1 text-sm text-gray-500">Access your donor portfolio, organization, or review queue</p>
      </div>
      <LoginForm />
    </div>
  );
}

