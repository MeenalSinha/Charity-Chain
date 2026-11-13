import type { Metadata } from 'next';
import './globals.css';
import { siteConfig } from '@/config/site';
import { QueryProvider } from '@/lib/api/providers';

export const metadata: Metadata = {
  title: {
    default: `${siteConfig.name} — Give with Evidence. Release with Confidence.`,
    template: `%s | ${siteConfig.name}`,
  },
  description: siteConfig.description,
  keywords: ['charity', 'donations', 'impact', 'transparency', 'ai verification', 'milestones'],
  authors: [{ name: 'CharityChain' }],
  creator: 'CharityChain',
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: siteConfig.url,
    title: siteConfig.name,
    description: siteConfig.description,
    siteName: siteConfig.name,
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className="font-sans min-h-full flex flex-col text-gray-900 bg-white antialiased">
        <QueryProvider>
          {children}
        </QueryProvider>
      </body>
    </html>
  );
}
