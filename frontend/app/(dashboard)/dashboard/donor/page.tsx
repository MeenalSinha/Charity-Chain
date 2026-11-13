import { getServerToken } from '@/lib/api/server-client';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { KPICard } from '@/components/ui/kpi-card';
import { StatusBadge } from '@/components/ui/status-badge';
import { formatCurrency, formatDate } from '@/lib/formatters';
import {
  Heart,
  History,
  Award,
  TrendingUp,
  FileText,
  Building,
  CheckCircle2,
  Clock,
  ArrowRight
} from 'lucide-react';
import Link from 'next/link';
import { redirect } from 'next/navigation';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

async function getDonorData() {
  const token = await getServerToken();
  if (!token) {
    redirect('/login?redirect=/dashboard/donor');
  }

  try {
    const [userRes, overviewRes, portfolioRes, journeyRes] = await Promise.all([
      fetch(`${BACKEND_URL}/api/v1/users/me`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
      }),
      fetch(`${BACKEND_URL}/api/v1/analytics/donor/overview`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
      }),
      fetch(`${BACKEND_URL}/api/v1/analytics/donor/portfolio`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
      }),
      fetch(`${BACKEND_URL}/api/v1/analytics/donor/journey`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
      }),
    ]);

    const user = userRes.ok ? (await userRes.json()).data : null;
    const overview = overviewRes.ok ? (await overviewRes.json()).data : null;
    const portfolio = portfolioRes.ok ? (await portfolioRes.json()).data : null;
    const journey = journeyRes.ok ? (await journeyRes.json()).data : [];

    return { user, overview, portfolio, journey };
  } catch {
    return { user: null, overview: null, portfolio: null, journey: [] };
  }
}

const navItems = [
  { href: '/dashboard/donor', label: 'Portfolio Overview', icon: <TrendingUp className="h-4 w-4" /> },
  { href: '/projects', label: 'Explore Projects', icon: <Heart className="h-4 w-4" /> },
  { href: '/transparency', label: 'Transparency', icon: <FileText className="h-4 w-4" /> },
];

export default async function DonorDashboardPage() {
  const { user, overview, portfolio, journey } = await getDonorData();

  return (
    <DashboardLayout navItems={navItems} title="Donor Portfolio" user={user}>
      <div className="space-y-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            Welcome back, {user?.full_name || 'Supporter'}
          </h1>
          <p className="text-sm text-gray-500">
            Track your verified contributions, milestones reached, and Proof-of-Impact certificates.
          </p>
        </div>

        {/* Financial KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="Total Captured"
            value={formatCurrency(overview?.total_captured_cents || 0)}
            subtitle="Verified via PayPal"
            icon={<Heart className="h-5 w-5" />}
          />
          <KPICard
            title="Net Impact Giving"
            value={formatCurrency(overview?.net_contributions_cents || 0)}
            subtitle="After any refunds"
            icon={<TrendingUp className="h-5 w-5" />}
          />
          <KPICard
            title="Supported Projects"
            value={overview?.supported_projects_count ?? 0}
            subtitle="Across organizations"
            icon={<Building className="h-5 w-5" />}
          />
          <KPICard
            title="Impact Certificates"
            value={overview?.certificates_count ?? 0}
            subtitle="Cryptographically signed"
            icon={<Award className="h-5 w-5" />}
          />
        </div>

        {/* Contribution to Impact Journey */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Clock className="h-5 w-5 text-blue-600" />
              Contribution-to-Impact Journeys
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!journey || journey.length === 0 ? (
              <div className="py-8 text-center text-sm text-gray-500">
                No active contribution journeys yet. Support a project to follow your impact!
                <div className="mt-3">
                  <Link href="/projects" className="text-blue-600 hover:underline font-medium">
                    Explore active projects →
                  </Link>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {journey.map((item: any, idx: number) => (
                  <div key={idx} className="p-4 rounded-lg border border-gray-200 bg-gray-50/50 space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="font-semibold text-sm text-gray-900">
                        {item.project_title || 'Project'} · {formatCurrency(item.donation_amount_cents || 0)}
                      </span>
                      <StatusBadge status={item.stage || 'PROCESSING'} />
                    </div>
                    {/* Visual Journey Lifecycle */}
                    <div className="flex items-center text-xs text-gray-500 overflow-x-auto py-2 gap-2">
                      <span className="font-medium text-emerald-600">Donation Captured</span>
                      <span>→</span>
                      <span className={item.milestone_started ? 'text-emerald-600 font-medium' : 'text-gray-400'}>
                        Milestone Active
                      </span>
                      <span>→</span>
                      <span className={item.evidence_submitted ? 'text-emerald-600 font-medium' : 'text-gray-400'}>
                        Evidence Submitted
                      </span>
                      <span>→</span>
                      <span className={item.dual_approved ? 'text-emerald-600 font-medium' : 'text-gray-400'}>
                        Dual-Key Approved
                      </span>
                      <span>→</span>
                      <span className={item.certificate_issued ? 'text-blue-600 font-bold' : 'text-gray-400'}>
                        Certificate Issued
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Donation History Table */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <History className="h-5 w-5 text-blue-600" />
              Recent Donations & Receipts
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!portfolio?.donations || portfolio.donations.length === 0 ? (
              <p className="text-sm text-gray-500 py-4 text-center">No recorded donations found.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 text-gray-500 text-xs">
                    <tr>
                      <th className="px-4 py-2 text-left">Project</th>
                      <th className="px-4 py-2 text-left">Amount</th>
                      <th className="px-4 py-2 text-left">Status</th>
                      <th className="px-4 py-2 text-left">Date</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {portfolio.donations.map((d: any) => (
                      <tr key={d.donation_id}>
                        <td className="px-4 py-3 font-medium text-gray-900">{d.project_title}</td>
                        <td className="px-4 py-3">{formatCurrency(d.amount_cents, d.currency)}</td>
                        <td className="px-4 py-3"><StatusBadge status={d.status} /></td>
                        <td className="px-4 py-3 text-gray-500">{formatDate(d.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}

