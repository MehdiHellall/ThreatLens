const apiBaseUrl = process.env.VITE_API_BASE_URL?.trim();

// Production uses Nginx's same-origin /api proxy by default. An explicit URL is
// still supported for standalone builds and must meet the checks below.
if (!apiBaseUrl || apiBaseUrl === "/api") {
  process.exit(0);
}

function isLoopbackHost(hostname) {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

try {
  const parsedUrl = new URL(apiBaseUrl);
  if (!["http:", "https:"].includes(parsedUrl.protocol)) {
    throw new Error("unsupported protocol");
  }
  if (parsedUrl.protocol === "http:" && !isLoopbackHost(parsedUrl.hostname)) {
    throw new Error("non-loopback http");
  }
  if (parsedUrl.pathname !== "/" || parsedUrl.search || parsedUrl.hash) {
    throw new Error("not an origin");
  }
} catch {
  console.error(
    "VITE_API_BASE_URL must be /api or an http(s) origin; non-loopback origins must use https.",
  );
  process.exit(1);
}
