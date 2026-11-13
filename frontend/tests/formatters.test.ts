import { describe, it, expect } from 'vitest';
import { formatCurrency, formatPercent, formatDate, formatRelativeTime, truncate } from '@/lib/formatters';

describe('Formatters', () => {
  describe('formatCurrency', () => {
    it('formats cents into dollar strings', () => {
      expect(formatCurrency(10050, 'USD')).toBe('$100.50');
      expect(formatCurrency(0, 'USD')).toBe('$0.00');
      expect(formatCurrency(500, 'USD')).toBe('$5.00');
    });

    it('handles negative or zero minor units safely', () => {
      expect(formatCurrency(0)).toBe('$0.00');
    });
  });

  describe('formatPercent', () => {
    it('formats float to percentage string', () => {
      expect(formatPercent(75.5)).toBe('75.5%');
      expect(formatPercent(100, 0)).toBe('100%');
    });
  });

  describe('formatDate', () => {
    it('formats ISO date strings', () => {
      const res = formatDate('2026-05-15T12:00:00Z');
      expect(res).toContain('2026');
      expect(res).toContain('May');
    });

    it('handles null/undefined gracefully', () => {
      expect(formatDate(null)).toBe('N/A');
      expect(formatDate(undefined)).toBe('N/A');
    });
  });

  describe('truncate', () => {
    it('truncates strings exceeding max length', () => {
      expect(truncate('Hello World', 5)).toBe('Hello…');
      expect(truncate('Short', 10)).toBe('Short');
    });
  });
});

