import Link from 'next/link';
import { notFound } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { StatusBadge } from '@/components/ui/status-badge';
import { formatCurrency, formatDate } from '@/lib/formatters';
import { Building2, Globe, ShieldCheck, ArrowLeft } from 'lucide-react';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

async function getOrgProfile(organizationId: string) {
  try {
    const res = await fetch(`${BACKEND_URL}/api/v1/transparency/organizations/${organizationId}`, {
      next: { revalidate: 60 },
    });
    if (!res.ok) return null;
    const json = await res.json();
    return json.data;
  } catch {
    return null;
  }
}

export default async function OrganizationPublicPage({
  params,
}: {
  params: Promise<{ organizationId: string }>;
}) {
  const { organizationId } = await params;
  const org = await getOrgProfile(organizationId);

  if (!org) {
    notFound();
  }

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      <Link href="/projects" className="inline-flex items-center text-sm text-gray-500 hover:text-blue-600 gap-1">
        <ArrowLeft className="h-4 w-4" /> Back to projects
      </Link>

      <div className="bg-white p-8 rounded-xl border border-gray-200 shadow-sm space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="p-4 bg-blue-50 text-blue-600 rounded-xl">
              <Building2 className="h-8 w-8" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">{org.name}</h1>
              {org.country && (
                <p className="text-sm text-gray-500 flex items-center gap-1 mt-0.5">
                  <Globe className="h-3.5 w-3.5" /> Country: {org.country}
                </p>
              )}
            </div>
          </div>
          <StatusBadge status={org.status} />
        </div>

        {org.mission_statement && (
          <div className="pt-4 border-t border-gray-100">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Mission Statement</h3>
            <p className="mt-1 text-sm text-gray-700 leading-relaxed">{org.mission_statement}</p>
          </div>
        )}
      </div>

      {/* Published Projects */}
      <div className="space-y-4">
        <h2 className="text-xl font-bold text-gray-900">Published Projects</h2>
        {!org.projects || org.projects.length === 0 ? (
          <p className="text-sm text-gray-500">No public projects available for this organization.</p>
        ) : (
          <div className="grid md:grid-cols-2 gap-6">
            {org.projects.map((p: any) => (
              <Card key={p.project_id}>
                <CardContent className="p-6 space-y-3">
                  <div className="flex justify-between items-start">
                    <h3 className="font-semibold text-gray-900">{p.title}</h3>
                    <StatusBadge status={p.status} />
                  </div>
                  <p className="text-sm text-gray-600 line-clamp-2">{p.description}</p>
                  <div className="flex justify-between items-center text-xs text-gray-500 pt-2 border-t border-gray-100">
                    <span>Target: {formatCurrency(p.target_amount_cents || 0)}</span>
                    <Link href={`/projects/${p.project_id}`} className="text-blue-600 hover:underline font-medium">
                      View Project →
                    </Link>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

