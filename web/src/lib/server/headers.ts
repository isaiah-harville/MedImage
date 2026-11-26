const READ_METHODS = new Set(['GET', 'HEAD']);

export function withPrivacyHeaders(method: string, response: Response): Response {
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('Referrer-Policy', 'no-referrer');
  if (!READ_METHODS.has(method)) response.headers.set('Cache-Control', 'no-store');
  return response;
}
