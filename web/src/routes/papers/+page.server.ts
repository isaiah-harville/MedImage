import { listPapers } from '#lib/server/papers.ts';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ locals }) => ({
  papers: await listPapers(locals.config.papersDir),
});
