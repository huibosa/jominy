// Injected by the Tauri shell at startup (desktop mode).
// In web/dev mode we use relative URLs so requests go through Vite's `/api`
// proxy regardless of which host the page was served from (localhost, LAN IP,
// etc.) — hardcoding `localhost:8000` would break for any non-local viewer.
const BASE: string =
  (window as unknown as { __JOMINY_API__?: string }).__JOMINY_API__ ?? "";

export { BASE };
