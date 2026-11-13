import Link from 'next/link';
import { notFound } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ProgressBar } from '@/components/ui/progress-bar';
import { StatusBadge } from '@/components/ui/status-badge';
import { formatCurrency, formatDate } from '@/lib/formatters';
import { Shield, CheckCircle, FileText, ArrowLeft, Heart } from 'lucide-react';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

async function getProjectDetails(projectId: string) {
  try {
    const res = await fetch(`${BACKEND_URL}/api/v1/transparency/projects/${projectId}`, {
      next: { revalidate: 30 },
    });
    if (!res.ok) return null;
    const json = await res.json();
    return json.data;
  } catch {
    return null;
  }
}

export default async function ProjectDetailPage({
  params,
}: {
  params: Promise<{ projectId: string }>;
}) {
  const { projectId } = await params;
  const project = await getProjectDetails(projectId);

  if (!project) {
    notFound();
  }

  const target = project.target_amount_cents || 1;
  const raised = project.raised_amount_cents || 0;
  const progressPercent = Math.min(100, Math.round((raised / target) * 100));

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      <Link href="/projects" className="inline-flex items-center text-sm text-gray-500 hover:text-blue-600 gap-1">
        <ArrowLeft className="h-4 w-4" /> Back to projects
      </Link>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Left 2 Columns: Project Overview and Milestones */}
        <div className="lg:col-span-2 space-y-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-semibold text-blue-600 uppercase tracking-wide">
                {project.category || 'General'}
              </span>
              <StatusBadge status={project.status} />
            </div>
            <h1 className="text-3xl font-bold text-gray-900">{project.title}</h1>
            <p className="mt-1 text-sm text-gray-500">
              By{' '}
              <Link href={`/organizations/${project.organization_id}`} className="text-blue-600 hover:underline">
                {project.organization_name}
              </Link>{' '}
              · Created {formatDate(project.created_at)}
            </p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>About the Project</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-700 leading-relaxed whitespace-pre-line text-sm">
                {project.description}
              </p>
            </CardContent>
          </Card>

          {/* Milestones & Evidence Status */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Milestones & Verification</CardTitle>
              <span className="text-xs text-gray-500">
                {project.milestones?.length || 0} sequential milestones
              </span>
            </CardHeader>
            <CardContent className="space-y-4">
              {!project.milestones || project.milestones.length === 0 ? (
                <p className="text-sm text-gray-500">No milestones registered yet.</p>
              ) : (
                project.milestones.map((m: any, idx: number) => (
                  <div key={m.milestone_id || idx} className="p-4 rounded-lg border border-gray-200 bg-gray-50/50 space-y-2">
                    <div className="flex items-start justify-between">
                      <div className="space-y-1">
                        <span className="text-xs font-bold text-blue-600 uppercase">
                          Milestone #{m.sequence_number || idx + 1}
                        </span>
                        <h4 className="font-semibold text-gray-900 text-sm">{m.title}</h4>
                      </div>
                      <StatusBadge status={m.status} />
                    </div>
                    {m.description && (
                      <p className="text-xs text-gray-600">{m.description}</p>
                    )}
                    <div className="flex justify-between items-center text-xs text-gray-500 pt-2 border-t border-gray-200">
                      <span>Target: {formatCurrency(m.target_amount_cents || 0)}</span>
                      <span>{m.evidence_count || 0} evidence files submitted</span>
                    </div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Funding Progress & Donation Action */}
        <div className="space-y-6">
          <Card className="sticky top-24">
            <CardHeader>
              <CardTitle className="text-xl">Support this Project</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="font-bold text-2xl text-gray-900">{formatCurrency(raised)}</span>
                  <span className="text-gray-500 text-sm self-end">of {formatCurrency(target)}</span>
                </div>
                <ProgressBar value={progressPercent} color="blue" showLabel />
              </div>

              <div className="rounded-lg bg-blue-50 p-4 border border-blue-100 text-xs text-blue-800 space-y-2">
                <div className="flex items-center gap-1.5 font-semibold">
                  <Shield className="h-4 w-4 text-blue-600 flex-shrink-0" />
                  Evidence-Gated Escrow
                </div>
                <p>
                  Donations are held safely and only released when the NGO submits verifiable evidence and dual reviewers approve.
                </p>
              </div>

              <div className="space-y-3">
                <Link href={`/login?redirect=/projects/${project.project_id}`}>
                  <Button className="w-full flex items-center justify-center gap-2" size="lg">
                    <Heart className="h-4 w-4" />
                    Donate to Project
                  </Button>
                </Link>
                <p className="text-center text-xs text-gray-500">
                  Secure payments powered by PayPal REST API
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

