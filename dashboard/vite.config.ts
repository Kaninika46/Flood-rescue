import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the dashboard calls /api/..., which Vite forwards to the FastAPI backend.
// For a build served elsewhere, set VITE_API_URL (e.g. http://localhost:8000) and CORS_ORIGINS.
export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 1400 }, // maplibre-gl 6 alone is ~1 MB
  // `npm run demo` serves the production build on http://localhost:4173 (localhost only); the
  // /api proxy below applies to it too (preview.proxy defaults to server.proxy).
  preview: { port: 4173, strictPort: true },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.API_TARGET ?? "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
