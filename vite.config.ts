import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';  // React plugin for Vite
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    host: "::",  // Ensures the app is accessible on all network interfaces
    port: 8080,  // Port number for the server
  },
  plugins: [react()],  // Use the React plugin
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),  // Resolve `@` to `./src` directory
    },
  },
  build: {
    outDir: 'dist',  // Specify the output directory for the build
  },
});
