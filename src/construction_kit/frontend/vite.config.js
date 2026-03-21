import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: '../public',  // Output to the Crystal server's static dir
    emptyOutDir: false,   // Don't delete existing files (components.json, svg/, etc.)
  },
  server: {
    proxy: {
      // Proxy API calls to Crystal server during development
      '/api': 'http://127.0.0.1:8081',
      '/ws': { target: 'ws://127.0.0.1:8081', ws: true },
    },
  },
})
