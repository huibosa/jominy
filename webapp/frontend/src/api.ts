// Injected by the Tauri shell at startup (desktop mode).
// Falls back to the Vite dev proxy base in web/dev mode.
const BASE: string =
  (window as unknown as { __JOMINY_API__?: string }).__JOMINY_API__ ??
  "http://localhost:8000";

export { BASE };
