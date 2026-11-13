export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000/api/v1';
export const APP_URL = process.env.NEXT_PUBLIC_APP_URL ?? 'http://localhost:3000';

export const ROUTES = {
  home: '/',
  login: '/login',
  register: '/register',
  verifyEmail: '/verify-email',
  forgotPassword: '/forgot-password',
  resetPassword: '/reset-password',
  projects: '/projects',
  organizations: '/organizations',
  transparency: '/transparency',
  impactMap: '/impact-map',
  certificates: '/certificates',
  donorDashboard: '/dashboard/donor',
  ngoDashboard: '/dashboard/ngo',
  reviewerDashboard: '/dashboard/reviewer',
  adminDashboard: '/dashboard/admin',
} as const;

export const PAGINATION_DEFAULTS = {
  pageSize: 20,
  maxPageSize: 100,
} as const;

export const SUPPORTED_CURRENCIES = ['USD', 'EUR', 'GBP', 'CAD', 'AUD'] as const;

