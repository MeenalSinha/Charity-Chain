import { test, expect } from '@playwright/test';

test.describe('Authentication Flows', () => {
  test('login page renders email and password fields', async ({ page }) => {
    await page.goto('/login');
    await expect(page.locator('input[type="email"]')).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
    await expect(page.locator('button[type="submit"]')).toContainText(/sign in/i);
  });

  test('login page provides links to register and password reset', async ({ page }) => {
    await page.goto('/login');
    const registerLink = page.locator('a[href="/register"]');
    const forgotLink = page.locator('a[href="/forgot-password"]');
    await expect(registerLink).toBeVisible();
    await expect(forgotLink).toBeVisible();
  });

  test('registration page renders required onboarding fields', async ({ page }) => {
    await page.goto('/register');
    await expect(page.locator('input[type="email"]')).toBeVisible();
    await expect(page.locator('button[type="submit"]')).toBeVisible();
  });
});
