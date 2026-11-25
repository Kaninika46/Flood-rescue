import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the dashboard calls /api/..., which Vite forwards to the FastAPI backend.
// For a build served elsewhere, set VITE_API_URL (e.g. http://localhost:8000) and CORS_ORIGINS.
export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 1200 }, // maplibre-gl alone is ~800 kB
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
