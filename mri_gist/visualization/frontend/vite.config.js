import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        secure: false,
      },
      // If we need to proxy static files that are served by FastAPI but not in public
      // But we copied everything to public, so it should be fine.
      '/static': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        // We might want to serve local static files first? 
        // Vite serves from public by default.
        // If a file is not found in public, it 404s.
        // If we want to fallback to backend static, we can proxy.
        // But for now, let's assume public has what we need.
      }
    }
  }
})
