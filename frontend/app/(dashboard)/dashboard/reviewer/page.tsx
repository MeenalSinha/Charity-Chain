import { getServerToken } from '@/lib/api/server-client';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { KPICard } from '@/components/ui/kpi-card';
import { StatusBadge } from '@/components/ui/status-badge';
import { Button } from '@/components/ui/button';
import { formatDate } from '@/lib/formatters';
import {
  CheckCircle2,
  Clock,
  AlertCircle,
  FileCheck,
  Shield,
  Eye,
  Bot
} from 'lucide-react';
import Link from 'next/link';
import { redirect } from 'next/navigation';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

async function getReviewerData() {
  const token = await getServerToken();
  if (!token) {
    redirect('/login?redirect=/dashboard/reviewer');
  }

  try {
    const [userRes, casesRes] = await Promise.all([
      fetch(`${BACKEND_URL}/api/v1/users/me`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
      }),
      fetch(`${BACKEND_URL}/api/v1/reviews/cases?limit=20`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
      }),
    ]);

    const user = userRes.ok ? (await userRes.json()).data : null;
    const cases = casesRes.ok ? (await casesRes.json()).data : [];

    return { user, cases };
  } catch {
    return { user: null, cases: [] };
  }
}

const navItems = [
  { href: '/dashboard/reviewer', label: 'Review Queue', icon: <FileCheck className="h-4 w-4" /> },
  { href: '/transparency', label: 'Transparency', icon: <Shield className="h-4 w-4" /> },
];

export default async function ReviewerDashboardPage() {
  const { user, cases } = await getReviewerData();

  return (
    <DashboardLayout navItems={navItems} title="Reviewer Operations" user={user}>
      <div className="space-y-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Review Queue & Dual-Key Approvals</h1>
          <p className="text-sm text-gray-500">
            Inspect evidence submissions, verify AI diagnostics, and provide independent human approval decisions.
          </p>
        </div>

        {/* Workload and Status KPIs */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <KPICard
            title="Assigned Cases"
            value={cases?.length || 0}
            subtitle="Under 5-task workload cap"
            icon={<Clock className="h-5 w-5" />}
          />
          <KPICard
            title="Approval Policy"
            value="Dual Independent"
            subtitle="Requires Slot 1 & Slot 2"
            icon={<Shield className="h-5 w-5" />}
          />
          <KPICard
            title="Blind Masking"
            value="Active"
            subtitle="Anti-anchoring enabled"
            icon={<Eye className="h-5 w-5" />}
          />
        </div>

        {/* Review Cases List */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <FileCheck className="h-5 w-5 text-blue-600" />
              Active Review Assignments
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!cases || cases.length === 0 ? (
              <div className="py-12 text-center text-sm text-gray-500">
                <CheckCircle2 className="h-10 w-10 text-emerald-500 mx-auto mb-2" />
                <p className="font-semibold text-gray-800">Your review queue is clear!</p>
                <p className="text-xs text-gray-400 mt-1">No pending milestone evidence requires decision at this time.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {cases.map((c: any) => (
                  <div key={c.case_id} className="p-4 rounded-lg border border-gray-200 bg-gray-50/50 space-y-3">
                    <div className="flex flex-wrap justify-between items-start gap-2">
                      <div>
                        <span className="text-xs font-semibold text-blue-600 uppercase">
                          Case #{c.case_id?.slice(0, 8)} · Round {c.round_number || 1}
                        </span>
                        <h4 className="font-semibold text-gray-900 text-sm mt-0.5">
                          {c.project_title || 'Milestone Verification'}
                        </h4>
                        <p className="text-xs text-gray-500">
                          Milestone: {c.milestone_title || 'Milestone'}
                        </p>
                      </div>
                      <StatusBadge status={c.status} />
                    </div>

                    <div className="flex flex-wrap items-center justify-between text-xs text-gray-600 pt-2 border-t border-gray-200 gap-2">
                      <div className="flex items-center gap-3">
                        <span className="flex items-center gap-1 text-blue-700">
                          <Bot className="h-3.5 w-3.5" /> AI Confidence: {c.ai_confidence ? `${(c.ai_confidence * 100).toFixed(0)}%` : 'Pending'}
                        </span>
                        <span>{c.evidence_count || 0} files attached</span>
                      </div>
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline">
                          Inspect Evidence
                        </Button>
                        <Button size="sm" variant="success">
                          Approve
                        </Button>
                      </div>
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

