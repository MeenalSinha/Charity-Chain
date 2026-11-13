'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { cn } from '@/lib/utilities';
import { logoutUser } from '@/lib/auth/client-auth';
import React, { useState } from 'react';
import { Menu, X, LogOut, Bell } from 'lucide-react';

export interface NavItem {
  href: string;
  label: string;
  icon?: React.ReactNode;
}

interface DashboardLayoutProps {
  children: React.ReactNode;
  navItems: NavItem[];
  title: string;
  user?: { email: string; full_name: string; role: string };
}

export function DashboardLayout({ children, navItems, title, user }: DashboardLayoutProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleLogout = async () => {
    try {
      await logoutUser();
      router.push('/login');
      router.refresh();
    } catch {
      router.push('/login');
    }
  };

  return (
    <div className="flex min-h-screen bg-gray-50">
      {/* Desktop Sidebar */}
      <aside className="hidden md:flex md:w-64 md:flex-col md:fixed md:inset-y-0 bg-white border-r border-gray-200 z-30">
        <div className="flex-1 flex flex-col min-h-0 pt-5 pb-4 overflow-y-auto">
          <div className="flex items-center justify-between px-6 mb-6">
            <Link href="/" className="text-xl font-bold text-blue-600">CharityChain</Link>
          </div>
          <p className="px-6 mb-3 text-xs font-semibold text-gray-400 uppercase tracking-wider">{title}</p>
          <nav className="flex-1 px-3 space-y-1" aria-label="Sidebar navigation">
            {navItems.map((item) => {
              const isActive = pathname === item.href || (item.href !== '/dashboard' && pathname.startsWith(item.href));
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-blue-50 text-blue-700 font-semibold'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  )}
                  aria-current={isActive ? 'page' : undefined}
                >
                  {item.icon && <span className="flex-shrink-0 text-current">{item.icon}</span>}
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
        <div className="flex-shrink-0 p-4 border-t border-gray-200">
          {user && (
            <div className="mb-3 px-2">
              <p className="text-sm font-medium text-gray-900 truncate">{user.full_name}</p>
              <p className="text-xs text-gray-500 truncate">{user.email}</p>
              <span className="inline-block mt-1 px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded">
                {user.role}
              </span>
            </div>
          )}
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-2 px-2 py-2 text-sm text-gray-600 hover:text-rose-600 transition-colors rounded hover:bg-rose-50"
          >
            <LogOut className="h-4 w-4" />
            <span>Sign out</span>
          </button>
        </div>
      </aside>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 flex md:hidden">
          <div className="fixed inset-0 bg-gray-600 bg-opacity-75" onClick={() => setMobileOpen(false)} />
          <div className="relative flex-1 flex flex-col max-w-xs w-full bg-white">
            <div className="absolute top-0 right-0 -mr-12 pt-2">
              <button
                className="ml-1 flex items-center justify-center h-10 w-10 rounded-full text-white"
                onClick={() => setMobileOpen(false)}
              >
                <X className="h-6 w-6" />
              </button>
            </div>
            <div className="flex-1 h-0 pt-5 pb-4 overflow-y-auto">
              <div className="flex items-center px-4 mb-5">
                <span className="text-xl font-bold text-blue-600">CharityChain</span>
              </div>
              <nav className="px-2 space-y-1">
                {navItems.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    onClick={() => setMobileOpen(false)}
                    className="flex items-center gap-3 px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:bg-gray-50"
                  >
                    {item.icon}
                    {item.label}
                  </Link>
                ))}
              </nav>
            </div>
            <div className="p-4 border-t border-gray-200">
              <button
                onClick={handleLogout}
                className="w-full flex items-center gap-2 text-sm text-rose-600"
              >
                <LogOut className="h-4 w-4" />
                Sign out
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="md:pl-64 flex flex-col flex-1 min-w-0">
        {/* Top bar on mobile */}
        <header className="md:hidden flex items-center justify-between h-16 px-4 bg-white border-b border-gray-200">
          <button
            onClick={() => setMobileOpen(true)}
            className="text-gray-500 hover:text-gray-700"
            aria-label="Open sidebar"
          >
            <Menu className="h-6 w-6" />
          </button>
          <span className="font-semibold text-gray-800">{title}</span>
          <div className="w-6" />
        </header>

        <main className="flex-1 p-6 md:p-8 max-w-7xl w-full mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
}

