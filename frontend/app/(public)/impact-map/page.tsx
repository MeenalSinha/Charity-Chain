import { ImpactMapClient } from '@/components/maps/ImpactMapClient';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

async function getMapMarkers() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/v1/transparency/map/markers`, {
      next: { revalidate: 60 },
    });
    if (!res.ok) return [];
    const json = await res.json();
    return json.data || [];
  } catch {
    return [];
  }
}

export default async function ImpactMapPage() {
  const markers = await getMapMarkers();

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Geographic Impact Map</h1>
        <p className="mt-2 text-gray-600">
          Explore where CharityChain initiatives are driving verified change across regions.
        </p>
      </div>

      <ImpactMapClient markers={markers} />
    </div>
  );
}

