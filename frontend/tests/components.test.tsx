import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { EmptyState } from '@/components/ui/empty-state';
import { ProgressBar } from '@/components/ui/progress-bar';

describe('UI Components', () => {
  it('renders Badge with correct label', () => {
    render(<Badge variant="success">Active</Badge>);
    expect(screen.getByText('Active')).toBeDefined();
  });

  it('renders Button and handles disabled attribute', () => {
    render(<Button disabled>Click Me</Button>);
    const btn = screen.getByRole('button', { name: 'Click Me' });
    expect(btn).toBeDefined();
    expect(btn).toHaveProperty('disabled', true);
  });

  it('renders EmptyState with title and description', () => {
    render(<EmptyState title="No Projects Found" description="Try broadening your search" />);
    expect(screen.getByText('No Projects Found')).toBeDefined();
    expect(screen.getByText('Try broadening your search')).toBeDefined();
  });

  it('renders ProgressBar with aria attributes', () => {
    render(<ProgressBar value={60} label="Funding progress" />);
    const bar = screen.getByRole('progressbar');
    expect(bar.getAttribute('aria-valuenow')).toBe('60');
    expect(bar.getAttribute('aria-valuemin')).toBe('0');
    expect(bar.getAttribute('aria-valuemax')).toBe('100');
  });
});

