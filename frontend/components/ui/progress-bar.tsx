import React from 'react';
import { cn } from '@/lib/utilities';

interface ProgressBarProps {
  value: number; // 0-100
  className?: string;
  label?: string;
  showLabel?: boolean;
  color?: 'blue' | 'green' | 'yellow' | 'red';
}

const colorMap = {
  blue: 'bg-blue-600',
  green: 'bg-emerald-500',
  yellow: 'bg-amber-500',
  red: 'bg-rose-500',
};

export function ProgressBar({ value, className, label, showLabel = false, color = 'blue' }: ProgressBarProps) {
  const clamped = Math.min(100, Math.max(0, value || 0));
  return (
    <div className={cn('space-y-1', className)}>
      {(label || showLabel) && (
        <div className="flex justify-between text-sm text-gray-600">
          {label && <span>{label}</span>}
          {showLabel && <span>{clamped.toFixed(0)}%</span>}
        </div>
      )}
      <div
        role="progressbar"
        aria-valuenow={clamped}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
        className="h-2 w-full overflow-hidden rounded-full bg-gray-200"
      >
        <div
          className={cn('h-full rounded-full transition-all duration-500', colorMap[color])}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

