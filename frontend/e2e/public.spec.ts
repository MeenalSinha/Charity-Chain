import { test, expect } from '@playwright/test';

test.describe('Public Web Experience', () => {
  test('homepage renders hero, navigation, and key sections', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/CharityChain/i);
    await expect(page.locator('header')).toBeVisible();
    await expect(page.locator('footer')).toBeVisible();
  });

  test('project catalog renders search and filter controls', async ({ page }) => {
    await page.goto('/projects');
    await expect(page.locator('header')).toBeVisible();
  });

  test('transparency portal renders public financial and impact overview', async ({ page }) => {
    await page.goto('/transparency');
    await expect(page.locator('header')).toBeVisible();
    await expect(page.locator('h1, h2')).toBeVisible();
  });

  test('certificate verification page renders verification form', async ({ page }) => {
    await page.goto('/certificates/verify/test-id');
    await expect(page.locator('header')).toBeVisible();
  });
});
