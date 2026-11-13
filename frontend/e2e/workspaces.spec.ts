import { test, expect } from '@playwright/test';

test.describe('Role Workspace Access Control', () => {
  test('unauthenticated donor dashboard access redirects to login', async ({ page }) => {
    await page.goto('/dashboard/donor');
    await expect(page).toHaveURL(/.*login/);
  });

  test('unauthenticated NGO dashboard access redirects to login', async ({ page }) => {
    await page.goto('/dashboard/ngo');
    await expect(page).toHaveURL(/.*login/);
  });

  test('unauthenticated reviewer dashboard access redirects to login', async ({ page }) => {
    await page.goto('/dashboard/reviewer');
    await expect(page).toHaveURL(/.*login/);
  });

  test('unauthenticated admin dashboard access redirects to login', async ({ page }) => {
    await page.goto('/dashboard/admin');
    await expect(page).toHaveURL(/.*login/);
  });
});

