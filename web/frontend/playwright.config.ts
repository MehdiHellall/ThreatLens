import { defineConfig, devices } from "@playwright/test";

const backendPort = Number(process.env.E2E_BACKEND_PORT ?? 8787);
const frontendPort = Number(process.env.E2E_FRONTEND_PORT ?? 5174);
const backendUrl = `http://127.0.0.1:${backendPort}`;
const frontendUrl = process.env.E2E_BASE_URL ?? `http://127.0.0.1:${frontendPort}`;
const usePreviewBuild = process.env.E2E_FRONTEND_MODE === "preview";
const useExistingServers = Boolean(process.env.E2E_BASE_URL);

const frontendServer = usePreviewBuild
  ? {
      command: `node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port ${frontendPort}`,
      url: frontendUrl,
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    }
  : {
      command: `node node_modules/vite/bin/vite.js --host 127.0.0.1 --port ${frontendPort}`,
      env: {
        VITE_API_PROXY_TARGET: backendUrl,
      },
      url: frontendUrl,
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    };

const webServers = useExistingServers
  ? []
  : [
      {
        command: "node scripts/start-backend.mjs",
        url: `${backendUrl}/v1/ready`,
        reuseExistingServer: !process.env.CI,
        timeout: 180_000,
      },
      frontendServer,
    ];

export default defineConfig({
  testDir: "./e2e",
  timeout: 180_000,
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  use: {
    baseURL: frontendUrl,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium-desktop",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } },
    },
    {
      name: "chromium-mobile",
      use: { ...devices["Pixel 5"] },
    },
  ],
  webServer: webServers,
});
