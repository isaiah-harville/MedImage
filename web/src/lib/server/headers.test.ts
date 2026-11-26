import { describe, expect, it } from 'vitest';
import { withPrivacyHeaders } from './headers.ts';

describe('withPrivacyHeaders', () => {
  it('marks a form action response, which carries the prediction, as no-store', () => {
    const response = withPrivacyHeaders('POST', new Response('prediction'));

    expect(response.headers.get('Cache-Control')).toBe('no-store');
    expect(response.headers.get('Referrer-Policy')).toBe('no-referrer');
  });

  it('leaves caching of page and image reads to the route', () => {
    const response = withPrivacyHeaders(
      'GET',
      new Response('png', { headers: { 'Cache-Control': 'public, max-age=3600' } }),
    );

    expect(response.headers.get('Cache-Control')).toBe('public, max-age=3600');
    expect(response.headers.get('Referrer-Policy')).toBe('no-referrer');
  });
});
