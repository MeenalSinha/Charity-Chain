export const AUTH_ENDPOINTS = {
  register: '/auth/register',
  login: '/auth/login',
  logout: '/auth/logout',
  logoutAll: '/auth/logout-all',
  refresh: '/auth/refresh',
  verifyEmail: '/auth/verify-email',
  resendVerification: '/auth/resend-verification',
  forgotPassword: '/auth/forgot-password',
  resetPassword: '/auth/reset-password',
  changePassword: '/auth/change-password',
  sessions: '/auth/sessions',
  revokeSession: (id: string) => `/auth/sessions/${id}/revoke`,
} as const;

export const USER_ENDPOINTS = {
  me: '/users/me',
  profile: '/users/me/profile',
} as const;

export const ORG_ENDPOINTS = {
  list: '/organizations',
  create: '/organizations',
  get: (id: string) => `/organizations/${id}`,
  update: (id: string) => `/organizations/${id}`,
  submitVerification: (id: string) => `/organizations/${id}/submit-verification`,
  uploadDocument: (id: string) => `/organizations/${id}/documents`,
  members: (id: string) => `/organizations/${id}/members`,
  invitations: (id: string) => `/organizations/${id}/invitations`,
} as const;

export const PROJECT_ENDPOINTS = {
  list: '/projects',
  create: '/projects',
  get: (id: string) => `/projects/${id}`,
  update: (id: string) => `/projects/${id}`,
  validate: (id: string) => `/projects/${id}/validate`,
  submit: (id: string) => `/projects/${id}/submit`,
  milestones: (id: string) => `/projects/${id}/milestones`,
  funding: (id: string) => `/projects/${id}/funding`,
} as const;

export const MILESTONE_ENDPOINTS = {
  get: (id: string) => `/milestones/${id}`,
  update: (id: string) => `/milestones/${id}`,
  evidence: (id: string) => `/milestones/${id}/evidence`,
} as const;

export const DONATION_ENDPOINTS = {
  list: '/donations',
  create: '/donations',
  get: (id: string) => `/donations/${id}`,
  capture: (id: string) => `/donations/${id}/capture`,
  status: (id: string) => `/donations/${id}/status`,
} as const;

export const EVIDENCE_ENDPOINTS = {
  get: (id: string) => `/evidence/${id}`,
  download: (id: string) => `/evidence/${id}/download`,
  integrity: (id: string) => `/evidence/${id}/integrity`,
  auditTrail: (id: string) => `/evidence/${id}/audit-trail`,
} as const;

export const ANALYTICS_ENDPOINTS = {
  ngoOverview: (orgId: string) => `/analytics/ngo/${orgId}/overview`,
  ngoFunding: (orgId: string) => `/analytics/ngo/${orgId}/funding`,
  ngoProjects: (orgId: string) => `/analytics/ngo/${orgId}/projects`,
  ngoEvidence: (orgId: string) => `/analytics/ngo/${orgId}/evidence`,
  ngoImpactReport: (orgId: string) => `/analytics/ngo/${orgId}/impact-report`,
  donorOverview: '/analytics/donor/overview',
  donorPortfolio: '/analytics/donor/portfolio',
  donorJourney: '/analytics/donor/journey',
  donorAnnualReport: '/analytics/donor/annual-report',
  adminOverview: '/analytics/admin/overview',
  adminFinancial: '/analytics/admin/financial',
  adminVerification: '/analytics/admin/verification',
  adminHealth: '/analytics/admin/health',
} as const;

export const TRANSPARENCY_ENDPOINTS = {
  overview: '/transparency/overview',
  projects: '/transparency/projects',
  projectDetails: (id: string) => `/transparency/projects/${id}`,
  organization: (id: string) => `/transparency/organizations/${id}`,
  mapMarkers: '/transparency/map/markers',
  density: '/transparency/map/density',
  methodology: '/transparency/methodology',
} as const;

export const CERTIFICATE_ENDPOINTS = {
  list: '/certificates',
  get: (id: string) => `/certificates/${id}`,
  verify: (id: string) => `/certificates/verify/${id}`,
  download: (id: string) => `/certificates/${id}/download`,
} as const;

export const NOTIFICATION_ENDPOINTS = {
  list: '/notifications',
  unreadCount: '/notifications/unread-count',
  markRead: (id: string) => `/notifications/${id}/read`,
  archive: (id: string) => `/notifications/${id}/archive`,
  preferences: '/notifications/preferences',
} as const;

