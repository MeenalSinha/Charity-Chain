import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function isServer(): boolean {
  return typeof window === 'undefined';
}

export function generateCorrelationId(): string {
  return `fe-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

