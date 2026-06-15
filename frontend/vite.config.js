import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api -> FastAPI backend so the browser hits one origin in dev.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})