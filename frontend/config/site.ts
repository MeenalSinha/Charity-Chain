export const siteConfig = {
  name: 'CharityChain',
  tagline: 'Give with Evidence. Release with Confidence.',
  description:
    'AI-powered charitable giving and impact verification platform. Track donations, verify evidence, and follow the journey from contribution to measurable impact.',
  url: process.env.NEXT_PUBLIC_APP_URL ?? 'http://localhost:3000',
  ogImage: '/og-image.png',
  links: {
    transparency: '/transparency',
    projects: '/projects',
    organizations: '/organizations',
    impactMap: '/impact-map',
    certificates: '/certificates',
  },
} as const;

