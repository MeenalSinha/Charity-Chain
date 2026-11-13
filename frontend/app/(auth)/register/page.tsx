import { Metadata } from 'next';
import { RegisterForm } from '@/components/forms/RegisterForm';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Create Account | CharityChain',
  robots: { index: false },
};

export default function RegisterPage() {
  return (
    <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10 border border-gray-100">
      <div className="text-center mb-6">
        <Link href="/" className="inline-block">
          <span className="text-2xl font-bold text-blue-600">CharityChain</span>
        </Link>
        <h1 className="mt-2 text-xl font-semibold text-gray-900">Create an account</h1>
        <p className="mt-1 text-sm text-gray-500">Join the evidence-backed charitable giving platform</p>
      </div>
      <RegisterForm />
    </div>
  );
}

