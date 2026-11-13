import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Shield, CheckCircle2, TrendingUp, Eye, Award, ArrowRight } from 'lucide-react';
import { formatCurrency } from '@/lib/formatters';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

async function getPublicOverview() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/v1/transparency/overview`, {
      next: { revalidate: 60 },
    });
    if (!res.ok) return null;
    const json = await res.json();
    return json.data;
  } catch {
    return null;
  }
}

async function getFeaturedProjects() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/v1/transparency/projects?limit=3`, {
      next: { revalidate: 60 },
    });
    if (!res.ok) return [];
    const json = await res.json();
    return json.data || [];
  } catch {
    return [];
  }
}

export default async function HomePage() {
  const overview = await getPublicOverview();
  const projects = await getFeaturedProjects();

  return (
    <div className="space-y-16 pb-16">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-b from-blue-50/60 to-white py-20 lg:py-28">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 text-center space-y-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-blue-100/80 px-4 py-1.5 text-xs font-semibold text-blue-800">
            <Shield className="h-4 w-4" />
            Next-Generation Charitable Accountability
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight text-gray-900 sm:text-6xl">
            Give with Evidence. <br className="hidden sm:inline" />
            <span className="text-blue-600">Release with Confidence.</span>
          </h1>
          <p className="mx-auto max-w-2xl text-lg text-gray-600">
            CharityChain is an AI-powered charitable giving platform. Donations are held securely, verified via computer vision and human review, and released only when milestones are completed.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <Link href="/projects">
              <Button size="lg" className="px-8 shadow-md">
                Explore Projects <ArrowRight className="h-4 w-4 ml-1" />
              </Button>
            </Link>
            <Link href="/register">
              <Button variant="outline" size="lg" className="px-8">
                Register Your Organization
              </Button>
            </Link>
            <Link href="/transparency">
              <Button variant="ghost" size="lg">
                View Transparency Portal
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Live Platform Statistics */}
      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <Card>
            <CardContent className="p-6 text-center">
              <p className="text-sm font-medium text-gray-500">Total Funds Captured</p>
              <p className="mt-2 text-3xl font-bold text-gray-900">
                {formatCurrency(overview?.total_captured_cents || 0)}
              </p>
              <p className="mt-1 text-xs text-emerald-600 font-medium">Reconciled in Real Time</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <p className="text-sm font-medium text-gray-500">Verified Organizations</p>
              <p className="mt-2 text-3xl font-bold text-gray-900">
                {overview?.verified_organizations_count ?? '0'}
              </p>
              <p className="mt-1 text-xs text-gray-500">Document-Verified</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <p className="text-sm font-medium text-gray-500">Published Projects</p>
              <p className="mt-2 text-3xl font-bold text-gray-900">
                {overview?.published_projects_count ?? '0'}
              </p>
              <p className="mt-1 text-xs text-gray-500">Active & Completed</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <p className="text-sm font-medium text-gray-500">Completed Milestones</p>
              <p className="mt-2 text-3xl font-bold text-gray-900">
                {overview?.completed_milestones_count ?? '0'}
              </p>
              <p className="mt-1 text-xs text-emerald-600 font-medium">Evidence-Verified</p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* How It Works */}
      <section className="bg-gray-50 py-16">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-12">
            <h2 className="text-3xl font-bold text-gray-900">How CharityChain Works</h2>
            <p className="mt-3 text-gray-600">
              A transparent, end-to-end impact lifecycle ensuring donor contributions translate directly into verified real-world results.
            </p>
          </div>
          <div className="grid md:grid-cols-4 gap-8">
            {[
              {
                step: '01',
                title: 'Targeted Donation',
                desc: 'Donors contribute funds to specific project milestones processed securely via PayPal with zero floating-point loss.',
                icon: <TrendingUp className="h-6 w-6 text-blue-600" />,
              },
              {
                step: '02',
                title: 'Evidence Upload',
                desc: 'NGOs execute work and submit photos, inspection reports, and EXIF geospatial documents to the tamper-evident archive.',
                icon: <Eye className="h-6 w-6 text-blue-600" />,
              },
              {
                step: '03',
                title: 'AI & Dual-Key Review',
                desc: 'Computer vision runs forensic checks (ELA, SSIM, YOLOv8) followed by two independent human reviewers.',
                icon: <CheckCircle2 className="h-6 w-6 text-blue-600" />,
              },
              {
                step: '04',
                title: 'Certificate & Release',
                desc: 'Upon dual approval, funds disburse to the NGO and an Ed25519-signed Proof-of-Impact certificate issues.',
                icon: <Award className="h-6 w-6 text-blue-600" />,
              },
            ].map(({ step, title, desc, icon }) => (
              <div key={step} className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm relative">
                <span className="text-xs font-bold text-blue-600 uppercase tracking-widest">{step}</span>
                <div className="mt-3 mb-4">{icon}</div>
                <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
                <p className="mt-2 text-sm text-gray-600">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Featured Projects */}
      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Featured Active Projects</h2>
            <p className="text-sm text-gray-500">Support verified initiatives with tangible milestones</p>
          </div>
          <Link href="/projects" className="text-sm font-semibold text-blue-600 hover:underline">
            View all projects →
          </Link>
        </div>

        {projects.length === 0 ? (
          <div className="text-center py-12 bg-gray-50 rounded-lg border border-dashed border-gray-300">
            <p className="text-gray-500 text-sm">No active public projects published yet.</p>
            <div className="mt-3">
              <Link href="/register" className="text-blue-600 text-sm font-medium hover:underline">
                Register as an NGO to launch a project
              </Link>
            </div>
          </div>
        ) : (
          <div className="grid md:grid-cols-3 gap-6">
            {projects.map((p: any) => (
              <Card key={p.project_id} className="hover:shadow-md transition-shadow flex flex-col">
                <CardContent className="p-6 flex-1 flex flex-col justify-between">
                  <div>
                    <span className="text-xs font-semibold text-blue-600 uppercase">{p.category || 'General'}</span>
                    <h3 className="text-lg font-semibold text-gray-900 mt-1 line-clamp-1">{p.title}</h3>
                    <p className="text-xs text-gray-500 mt-0.5">{p.organization_name}</p>
                    <p className="text-sm text-gray-600 mt-3 line-clamp-3">{p.description}</p>
                  </div>
                  <div className="mt-6 pt-4 border-t border-gray-100 space-y-3">
                    <div className="flex justify-between text-xs text-gray-600">
                      <span>Raised: {formatCurrency(p.raised_amount_cents || 0)}</span>
                      <span>Target: {formatCurrency(p.target_amount_cents || 0)}</span>
                    </div>
                    <Link href={`/projects/${p.project_id}`}>
                      <Button variant="outline" size="sm" className="w-full">
                        View Project Details
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </section>

      {/* Trust & Transparency */}
      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="rounded-2xl bg-blue-900 text-white p-8 md:p-12 shadow-xl">
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div className="space-y-4">
              <span className="text-xs uppercase font-semibold text-blue-300 tracking-wider">Uncompromising Integrity</span>
              <h2 className="text-3xl font-bold">Cryptographically Verifiable Impact</h2>
              <p className="text-blue-100 text-sm leading-relaxed">
                CharityChain combines integer minor-unit financial accounting, double-entry general ledgers, authentic EXIF GPS parsing, and Ed25519 asymmetric signatures. Anyone can independently verify certificates using public RFC 7517 JWKS discovery endpoints.
              </p>
              <div className="pt-2 flex flex-wrap gap-4">
                <Link href="/transparency">
                  <Button variant="default" className="bg-white text-blue-900 hover:bg-blue-50">
                    Explore Transparency Data
                  </Button>
                </Link>
                <Link href="/certificates">
                  <Button variant="outline" className="border-blue-400 text-white hover:bg-blue-800">
                    Verify a Certificate
                  </Button>
                </Link>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 text-center">
              <div className="bg-blue-800/60 p-4 rounded-xl border border-blue-700">
                <p className="text-2xl font-bold">100%</p>
                <p className="text-xs text-blue-200 mt-1">Integer-Based Math</p>
              </div>
              <div className="bg-blue-800/60 p-4 rounded-xl border border-blue-700">
                <p className="text-2xl font-bold">Ed25519</p>
                <p className="text-xs text-blue-200 mt-1">Digital Signatures</p>
              </div>
              <div className="bg-blue-800/60 p-4 rounded-xl border border-blue-700">
                <p className="text-2xl font-bold">Dual-Key</p>
                <p className="text-xs text-blue-200 mt-1">Independent Review</p>
              </div>
              <div className="bg-blue-800/60 p-4 rounded-xl border border-blue-700">
                <p className="text-2xl font-bold">3-Way</p>
                <p className="text-xs text-blue-200 mt-1">Reconciliation</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

