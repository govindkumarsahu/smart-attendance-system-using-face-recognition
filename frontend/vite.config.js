import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/login': 'http://localhost:8000',
      '/logout': 'http://localhost:8000',
      '/register': 'http://localhost:8000',
      '/train': 'http://localhost:8000',
      '/attendance': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/reset': 'http://localhost:8000',
      '/student_image': 'http://localhost:8000',
      '/export-csv': 'http://localhost:8000',
      '/delete_student': 'http://localhost:8000',
      '/edit_student': 'http://localhost:8000',
    },
  },
})
