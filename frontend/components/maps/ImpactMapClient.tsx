'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { MapPin, Globe } from 'lucide-react';
import Link from 'next/link';

interface MapMarker {
  project_id: string;
  title: string;
  organization_name?: string;
  category?: string;
  lat: number;
  lng: number;
  is_obfuscated?: boolean;
}

export function ImpactMapClient({ markers }: { markers: MapMarker[] }) {
  const [selectedMarker, setSelectedMarker] = useState<MapMarker | null>(null);

  return (
    <div className="grid lg:grid-cols-3 gap-6">
      {/* Visual Map Representation */}
      <div className="lg:col-span-2 bg-slate-900 rounded-xl overflow-hidden min-h-[500px] flex flex-col justify-between p-6 relative shadow-inner">
        <div className="flex justify-between items-start z-10">
          <div className="bg-slate-800/80 backdrop-blur px-3 py-1.5 rounded-lg border border-slate-700 text-xs text-slate-300 flex items-center gap-2">
            <Globe className="h-4 w-4 text-blue-400" />
            <span>Interactive Geospatial Map (Privacy Preserved)</span>
          </div>
          <span className="bg-blue-600/80 text-white text-xs px-2.5 py-1 rounded-full font-medium">
            {markers.length} Project Markers
          </span>
        </div>

        {/* Visual Map Grid Canvas */}
        <div className="grid grid-cols-6 gap-3 py-12 px-4 z-0">
          {markers.slice(0, 12).map((m, i) => (
            <button
              key={m.project_id || i}
              onClick={() => setSelectedMarker(m)}
              className="group flex flex-col items-center justify-center p-3 rounded-lg bg-slate-800/60 hover:bg-blue-600/40 border border-slate-700 hover:border-blue-500 transition-all text-center"
            >
              <MapPin className="h-5 w-5 text-blue-400 group-hover:text-blue-300" />
              <span className="text-[10px] text-slate-300 mt-1 truncate max-w-full">
                {m.title || `Project ${i + 1}`}
              </span>
            </button>
          ))}
        </div>

        <div className="bg-slate-800/60 p-3 rounded-lg border border-slate-700 text-xs text-slate-400">
          💡 Sensitive coordinates are automatically coarsened to regional centroids (~11.1 km radius) to protect on-the-ground beneficiaries.
        </div>
      </div>

      {/* Marker Detail Sidebar */}
      <div className="space-y-4">
        <h2 className="text-lg font-bold text-gray-900">Project Locations</h2>
        {selectedMarker ? (
          <Card>
            <CardContent className="p-6 space-y-4">
              <span className="text-xs font-semibold text-blue-600 uppercase">
                {selectedMarker.category || 'Initiative'}
              </span>
              <h3 className="text-lg font-bold text-gray-900">{selectedMarker.title}</h3>
              {selectedMarker.organization_name && (
                <p className="text-xs text-gray-500">By {selectedMarker.organization_name}</p>
              )}
              <div className="text-xs text-gray-600 space-y-1 bg-gray-50 p-3 rounded border">
                <p>Coordinates: {selectedMarker.lat.toFixed(2)}°, {selectedMarker.lng.toFixed(2)}°</p>
                {selectedMarker.is_obfuscated && (
                  <p className="text-amber-600 font-medium">Obfuscated for location privacy</p>
                )}
              </div>
              <Link href={`/projects/${selectedMarker.project_id}`} className="block">
                <button className="w-full py-2 px-4 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-700 transition">
                  View Full Project
                </button>
              </Link>
            </CardContent>
          </Card>
        ) : (
          <div className="p-6 bg-white rounded-lg border border-gray-200 text-center text-sm text-gray-500">
            Select a project marker from the map to view coordinates, category, and details.
          </div>
        )}
      </div>
    </div>
  );
}

