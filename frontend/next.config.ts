import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  output: (process.env.BUILD_STANDALONE === 'true' || process.env.NEXT_OUTPUT === 'standalone') ? 'standalone' : undefined,
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/**',
      },
    ],
  },
  async rewrites() {
    const backendUrl = process.env.BACKEND_URL ?? 'http://localhost:8000';
    return [
      {
        source: '/api/backend/:path*',
        destination: `${backendUrl}/api/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
