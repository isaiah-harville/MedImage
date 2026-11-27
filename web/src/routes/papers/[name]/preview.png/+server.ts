import { error } from '@sveltejs/kit';
import { readPaperFile } from '#lib/server/papers.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ locals, params }) => {
  const bytes = await readPaperFile(locals.config.papersDir, params.name, 'preview');
  if (!bytes) error(404, 'There is no preview for that paper.');
  return new Response(bytes, {
    headers: { 'content-type': 'image/png', 'cache-control': 'public, max-age=3600' },
  });
};
