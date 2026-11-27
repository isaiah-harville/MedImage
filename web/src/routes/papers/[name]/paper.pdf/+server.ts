import { error } from '@sveltejs/kit';
import { readPaperFile } from '#lib/server/papers.ts';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ locals, params }) => {
  const bytes = await readPaperFile(locals.config.papersDir, params.name, 'pdf');
  if (!bytes) error(404, 'There is no paper with that name.');
  return new Response(bytes, {
    headers: {
      'content-type': 'application/pdf',
      'content-disposition': `inline; filename="${params.name}.pdf"`,
      'cache-control': 'public, max-age=3600',
    },
  });
};
