const READ_METHODS = new Set(['GET', 'HEAD']);

export function withPrivacyHeaders(method: string, response: Response): Response {
  response.headers.set('X-Content-Type-Options', 'nosniff');
  // Not no-referrer: under it browsers send `Origin: null` on same-origin form posts, which
  // SvelteKit's CSRF check rejects. same-origin still sends nothing to other sites.
  response.headers.set('Referrer-Policy', 'same-origin');
  if (!READ_METHODS.has(method)) response.headers.set('Cache-Control', 'no-store');
  return response;
}
