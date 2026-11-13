import Link from 'next/link';

export function PublicFooter() {
  return (
    <footer className="bg-gray-900 text-gray-300">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          <div className="col-span-2 md:col-span-1">
            <span className="text-xl font-bold text-white">CharityChain</span>
            <p className="mt-3 text-sm text-gray-400">Give with Evidence. Release with Confidence.</p>
          </div>
          {[
            {
              title: 'Platform',
              links: [
                { href: '/projects', label: 'Explore Projects' },
                { href: '/organizations', label: 'Organizations' },
                { href: '/transparency', label: 'Transparency' },
                { href: '/impact-map', label: 'Impact Map' },
              ],
            },
            {
              title: 'Verification',
              links: [
                { href: '/certificates', label: 'Certificate Verification' },
                { href: '/transparency', label: 'Methodology' },
              ],
            },
            {
              title: 'Account',
              links: [
                { href: '/register', label: 'Create Account' },
                { href: '/login', label: 'Sign In' },
              ],
            },
          ].map(({ title, links }) => (
            <div key={title}>
              <h3 className="text-sm font-semibold text-white">{title}</h3>
              <ul className="mt-4 space-y-2">
                {links.map(({ href, label }) => (
                  <li key={href}>
                    <Link href={href} className="text-sm text-gray-400 hover:text-white transition-colors">
                      {label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <div className="mt-8 border-t border-gray-800 pt-8 text-xs text-gray-500">
          <p>© {new Date().getFullYear()} CharityChain. AI-powered charitable giving and impact verification.</p>
        </div>
      </div>
    </footer>
  );
}

