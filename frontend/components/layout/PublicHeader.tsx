'use client';

import Link from 'next/link';
import { useState } from 'react';
import { Menu, X } from 'lucide-react';

export function PublicHeader() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 bg-white border-b border-gray-200">
      <nav className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8" aria-label="Main navigation">
        <div className="flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <span className="text-xl font-bold text-blue-600">CharityChain</span>
          </Link>
          <div className="hidden md:flex items-center gap-6 text-sm font-medium text-gray-600">
            <Link href="/projects" className="hover:text-blue-600 transition-colors">Explore Projects</Link>
            <Link href="/organizations" className="hover:text-blue-600 transition-colors">Organizations</Link>
            <Link href="/transparency" className="hover:text-blue-600 transition-colors">Transparency</Link>
            <Link href="/impact-map" className="hover:text-blue-600 transition-colors">Impact Map</Link>
          </div>
          <div className="hidden md:flex items-center gap-3">
            <Link
              href="/login"
              className="text-sm font-medium text-gray-600 hover:text-blue-600 transition-colors"
            >
              Sign in
            </Link>
            <Link
              href="/register"
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 transition-colors"
            >
              Get started
            </Link>
          </div>
          <div className="flex md:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="text-gray-500 hover:text-gray-700 p-2"
              aria-label="Toggle menu"
            >
              {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>
        {isOpen && (
          <div className="md:hidden py-4 border-t border-gray-100 space-y-3">
            <Link href="/projects" className="block text-sm font-medium text-gray-600 hover:text-blue-600">Explore Projects</Link>
            <Link href="/organizations" className="block text-sm font-medium text-gray-600 hover:text-blue-600">Organizations</Link>
            <Link href="/transparency" className="block text-sm font-medium text-gray-600 hover:text-blue-600">Transparency</Link>
            <Link href="/impact-map" className="block text-sm font-medium text-gray-600 hover:text-blue-600">Impact Map</Link>
            <div className="pt-2 border-t border-gray-100 flex flex-col gap-2">
              <Link href="/login" className="text-sm font-medium text-gray-600 hover:text-blue-600">Sign in</Link>
              <Link href="/register" className="text-sm font-medium text-blue-600">Get started</Link>
            </div>
          </div>
        )}
      </nav>
    </header>
  );
}

