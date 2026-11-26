import { error } from '@sveltejs/kit';
import { ApiError } from '#lib/server/api.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ locals, params }) => {
  let upstream: Response;
  try {
    upstream = await locals.api.demoImage(params.id, params.sample);
  } catch (cause) {
    if (!(cause instanceof ApiError) || cause.status !== 404) throw cause;
    error(404, 'No such demo image.');
  }
  return new Response(upstream.body, {
    headers: { 'Content-Type': 'image/png', 'Cache-Control': 'public, max-age=3600' },
  });
};
