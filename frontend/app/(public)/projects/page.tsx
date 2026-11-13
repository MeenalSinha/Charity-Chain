import Link from 'next/link';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ProgressBar } from '@/components/ui/progress-bar';
import { formatCurrency } from '@/lib/formatters';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

interface SearchParams {
  category?: string;
  search?: string;
}

async function getProjects(params: SearchParams) {
  try {
    const query = new URLSearchParams();
    if (params.category) query.set('category', params.category);
    query.set('limit', '50');

    const res = await fetch(`${BACKEND_URL}/api/v1/transparency/projects?${query.toString()}`, {
      next: { revalidate: 30 },
    });
    if (!res.ok) return [];
    const json = await res.json();
    let data = json.data || [];
    if (params.search) {
      const q = params.search.toLowerCase();
      data = data.filter((p: any) =>
        p.title?.toLowerCase().includes(q) ||
        p.organization_name?.toLowerCase().includes(q) ||
        p.description?.toLowerCase().includes(q)
      );
    }
    return data;
  } catch {
    return [];
  }
}

export default async function ProjectsPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const params = await searchParams;
  const projects = await getProjects(params);

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Explore Charitable Projects</h1>
        <p className="mt-2 text-gray-600">
          Browse verified initiatives with measurable milestones and real-time evidence verification.
        </p>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 flex flex-wrap gap-4 items-center justify-between">
        <div className="flex flex-wrap gap-2">
          {['All', 'Education', 'Health', 'Environment', 'Poverty', 'Disaster Relief'].map((cat) => {
            const isAll = cat === 'All';
            const isActive = isAll ? !params.category : params.category === cat;
            return (
              <Link
                key={cat}
                href={isAll ? '/projects' : `/projects?category=${encodeURIComponent(cat)}`}
                className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-100'
                }`}
              >
                {cat}
              </Link>
            );
          })}
        </div>
      </div>

      {/* Projects Grid */}
      {projects.length === 0 ? (
        <div className="text-center py-16 bg-white rounded-lg border border-gray-200">
          <p className="text-gray-500">No projects found matching the selected criteria.</p>
          <div className="mt-4">
            <Link href="/projects">
              <Button variant="outline" size="sm">
                Clear Filters
              </Button>
            </Link>
          </div>
        </div>
      ) : (
        <div className="grid md:grid-cols-3 gap-6">
          {projects.map((p: any) => {
            const target = p.target_amount_cents || 1;
            const raised = p.raised_amount_cents || 0;
            const percent = Math.min(100, Math.round((raised / target) * 100));

            return (
              <Card key={p.project_id} className="hover:shadow-md transition-shadow flex flex-col justify-between">
                <CardContent className="p-6 flex-1 flex flex-col justify-between space-y-4">
                  <div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-blue-600 uppercase">
                        {p.category || 'General'}
                      </span>
                      <span className="text-xs bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded font-medium">
                        {p.status}
                      </span>
                    </div>
                    <h2 className="text-lg font-semibold text-gray-900 mt-2 line-clamp-1">{p.title}</h2>
                    <p className="text-xs text-gray-500 mt-0.5">{p.organization_name}</p>
                    <p className="text-sm text-gray-600 mt-3 line-clamp-3">{p.description}</p>
                  </div>

                  <div className="space-y-3 pt-3 border-t border-gray-100">
                    <ProgressBar value={percent} showLabel label="Funding Progress" color="blue" />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Raised: {formatCurrency(raised)}</span>
                      <span>Goal: {formatCurrency(target)}</span>
                    </div>
                    <div className="pt-2">
                      <Link href={`/projects/${p.project_id}`}>
                        <Button variant="default" size="sm" className="w-full">
                          View Project & Donate
                        </Button>
                      </Link>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}

