import React from 'react';
import { Card, CardContent } from './card';
import { cn } from '@/lib/utilities';

interface KPICardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: { value: number; label: string };
  icon?: React.ReactNode;
  className?: string;
}

export function KPICard({ title, value, subtitle, trend, icon, className }: KPICardProps) {
  return (
    <Card className={cn('', className)}>
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="space-y-1 flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-500 truncate">{title}</p>
            <p className="text-2xl font-bold text-gray-900">{value}</p>
            {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
            {trend && (
              <p className={cn('text-sm', trend.value >= 0 ? 'text-emerald-600' : 'text-rose-600')}>
                {trend.value >= 0 ? '↑' : '↓'} {Math.abs(trend.value)}% {trend.label}
              </p>
            )}
          </div>
          {icon && (
            <div className="ml-4 p-3 bg-blue-50 rounded-lg text-blue-600 flex-shrink-0">
              {icon}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

