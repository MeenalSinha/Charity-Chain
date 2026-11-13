import { getServerToken } from '@/lib/api/server-client';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { KPICard } from '@/components/ui/kpi-card';
import { StatusBadge } from '@/components/ui/status-badge';
import { Button } from '@/components/ui/button';
import { formatCurrency } from '@/lib/formatters';
import {
  Building2,
  FolderPlus,
  Coins,
  CheckSquare,
  FileUp,
  TrendingUp,
  Users
} from 'lucide-react';
import Link from 'next/link';
import { redirect } from 'next/navigation';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

async function getNgoData() {
  const token = await getServerToken();
  if (!token) {
    redirect('/login?redirect=/dashboard/ngo');
  }

  try {
    const [userRes, orgsRes] = await Promise.all([
      fetch(`${BACKEND_URL}/api/v1/users/me`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
      }),
      fetch(`${BACKEND_URL}/api/v1/organizations`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
      }),
    ]);

    const user = userRes.ok ? (await userRes.json()).data : null;
    const orgs = orgsRes.ok ? (await orgsRes.json()).data : [];
    const activeOrg = orgs.length > 0 ? orgs[0] : null;

    let overview = null;
    let funding = null;

    if (activeOrg) {
      const [overviewRes, fundingRes] = await Promise.all([
        fetch(`${BACKEND_URL}/api/v1/analytics/ngo/${activeOrg.id}/overview`, {
          headers: { Authorization: `Bearer ${token}` },
          cache: 'no-store',
        }),
        fetch(`${BACKEND_URL}/api/v1/analytics/ngo/${activeOrg.id}/funding`, {
          headers: { Authorization: `Bearer ${token}` },
          cache: 'no-store',
        }),
      ]);
      overview = overviewRes.ok ? (await overviewRes.json()).data : null;
      funding = fundingRes.ok ? (await fundingRes.json()).data : null;
    }

    return { user, activeOrg, overview, funding };
  } catch {
    return { user: null, activeOrg: null, overview: null, funding: null };
  }
}

const navItems = [
  { href: '/dashboard/ngo', label: 'Organization Overview', icon: <Building2 className="h-4 w-4" /> },
  { href: '/projects', label: 'Public Projects', icon: <Coins className="h-4 w-4" /> },
  { href: '/transparency', label: 'Transparency', icon: <TrendingUp className="h-4 w-4" /> },
];

export default async function NgoDashboardPage() {
  const { user, activeOrg, overview, funding } = await getNgoData();

  if (!activeOrg) {
    return (
      <DashboardLayout navItems={navItems} title="NGO Workspace" user={user}>
        <div className="max-w-xl mx-auto py-12 text-center space-y-4">
          <Building2 className="h-12 w-12 text-blue-600 mx-auto" />
          <h1 className="text-2xl font-bold text-gray-900">Register Your Organization</h1>
          <p className="text-sm text-gray-600">
            You don't have an affiliated organization yet. Register your legal NGO to unlock milestone project management and fund disbursement.
          </p>
          <div className="pt-2">
            <Link href="/register">
              <Button>Create Organization Profile</Button>
            </Link>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout navItems={navItems} title="NGO Operations" user={user}>
      <div className="space-y-8">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-bold text-gray-900">{activeOrg.name}</h1>
              <StatusBadge status={activeOrg.status} />
            </div>
            <p className="text-sm text-gray-500 mt-1">
              Manage projects, upload evidence for completed milestones, and track donor disbursements.
            </p>
          </div>
          <div className="flex gap-2">
            <Button size="sm">
              <FolderPlus className="h-4 w-4 mr-1.5" />
              New Project
            </Button>
          </div>
        </div>

        {/* Operational Metrics */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="Total Raised"
            value={formatCurrency(funding?.total_raised_cents || 0)}
            subtitle="Across all milestones"
            icon={<Coins className="h-5 w-5" />}
          />
          <KPICard
            title="Total Donations"
            value={funding?.total_donations ?? 0}
            subtitle="From unique contributors"
            icon={<Users className="h-5 w-5" />}
          />
          <KPICard
            title="Active Projects"
            value={overview?.projects?.active ?? 0}
            subtitle="In progress"
            icon={<Building2 className="h-5 w-5" />}
          />
          <KPICard
            title="Completed Milestones"
            value={overview?.milestones?.completed ?? 0}
            subtitle="Verified by reviewers"
            icon={<CheckSquare className="h-5 w-5" />}
          />
        </div>

        {/* Project Performance & Evidence Submission Center */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <FileUp className="h-5 w-5 text-blue-600" />
              Evidence Submission & Milestone Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!funding?.funding_by_project || funding.funding_by_project.length === 0 ? (
              <p className="text-sm text-gray-500 py-6 text-center">
                No active projects found. Create a project to start receiving milestone funding.
              </p>
            ) : (
              <div className="space-y-4">
                {funding.funding_by_project.map((p: any) => (
                  <div key={p.project_id} className="p-4 rounded-lg border border-gray-200 bg-gray-50/50 flex flex-wrap justify-between items-center gap-4">
                    <div>
                      <h4 className="font-semibold text-gray-900 text-sm">{p.title}</h4>
                      <p className="text-xs text-gray-500 mt-0.5">
                        Raised {formatCurrency(p.raised_amount_cents || 0)} · {p.donation_count || 0} donations
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Link href={`/projects/${p.project_id}`}>
                        <Button variant="outline" size="sm">
                          View Project
                        </Button>
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}

