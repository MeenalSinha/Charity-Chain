import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { formatCurrency } from '@/lib/formatters';
import { Shield, BookOpen, CheckCircle, Database } from 'lucide-react';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

async function getTransparencyData() {
  try {
    const [overviewRes, methodologyRes] = await Promise.all([
      fetch(`${BACKEND_URL}/api/v1/transparency/overview`, { next: { revalidate: 60 } }),
      fetch(`${BACKEND_URL}/api/v1/transparency/methodology`, { next: { revalidate: 300 } }),
    ]);

    const overview = overviewRes.ok ? (await overviewRes.json()).data : null;
    const methodology = methodologyRes.ok ? (await methodologyRes.json()).data : null;

    return { overview, methodology };
  } catch {
    return { overview: null, methodology: null };
  }
}

export default async function TransparencyPage() {
  const { overview, methodology } = await getTransparencyData();

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      <div className="text-center max-w-3xl mx-auto space-y-4">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-100 text-blue-800 text-xs font-semibold">
          <Shield className="h-3.5 w-3.5" />
          Public Accountability Portal
        </div>
        <h1 className="text-4xl font-bold text-gray-900">CharityChain Transparency</h1>
        <p className="text-gray-600 text-base">
          Real-time, unauthenticated verification data directly from PostgreSQL and double-entry ledger records. Zero fabricated statistics.
        </p>
      </div>

      {/* Global Indicators */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6 text-center">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">Total Captured Funds</p>
            <p className="mt-2 text-3xl font-bold text-gray-900">{formatCurrency(overview?.total_captured_cents || 0)}</p>
            <p className="mt-1 text-xs text-emerald-600 font-medium">Reconciled Cents</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">Verified Organizations</p>
            <p className="mt-2 text-3xl font-bold text-gray-900">{overview?.verified_organizations_count ?? '0'}</p>
            <p className="mt-1 text-xs text-gray-500">Legal Document Audited</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">Completed Milestones</p>
            <p className="mt-2 text-3xl font-bold text-gray-900">{overview?.completed_milestones_count ?? '0'}</p>
            <p className="mt-1 text-xs text-emerald-600 font-medium">Dual-Key Verified</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">Active Projects</p>
            <p className="mt-2 text-3xl font-bold text-gray-900">{overview?.active_projects_count ?? '0'}</p>
            <p className="mt-1 text-xs text-gray-500">In Execution</p>
          </CardContent>
        </Card>
      </div>

      {/* Calculation Methodology */}
      <div className="space-y-6" id="methodology">
        <div className="flex items-center gap-2">
          <BookOpen className="h-6 w-6 text-blue-600" />
          <h2 className="text-2xl font-bold text-gray-900">Calculation Methodology & Governance</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Database className="h-4 w-4 text-blue-600" />
                Integer Minor-Unit Financial Integrity
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-600 space-y-2">
              <p>
                All balances, targets, and donations are calculated strictly in integer minor units (e.g. cents). Floating-point conversions are prohibited to prevent IEEE 754 precision drift.
              </p>
              <p>
                Financial figures reconcile continuously against the double-entry general ledger where Debits equal Credits.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-emerald-600" />
                Dual-Key Human Review
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-600 space-y-2">
              <p>
                AI verification (YOLOv8, ELA, SSIM) serves as an automated advisor. Milestone completion requires sign-off by two independent human reviewers with conflict-of-interest enforcement.
              </p>
              <p>
                Blind review masking prevents cognitive anchoring between Slot 1 and Slot 2 reviewers.
              </p>
            </CardContent>
          </Card>
        </div>

        {methodology && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Official Methodology Statement</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xs text-gray-600 space-y-2 font-mono bg-gray-50 p-4 rounded border">
                <p><strong>Version:</strong> {methodology.version ?? 'Phase 11'}</p>
                <p><strong>Authority:</strong> {methodology.authoritative_source ?? 'PostgreSQL'}</p>
                <p><strong>Standards:</strong> RFC 8785 Canonical JSON, Ed25519 Signatures, RFC 7517 JWKS</p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

