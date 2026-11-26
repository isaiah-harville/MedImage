import type { ModelSummary } from '#lib/contract.ts';
import { ApiError } from '#lib/server/api.ts';
import type { LayoutServerLoad } from './$types';

type Catalog = { kind: 'ready'; models: ModelSummary[] } | { kind: 'unavailable' };

export const load: LayoutServerLoad = async ({ locals }) => {
  let catalog: Catalog;
  try {
    catalog = { kind: 'ready', models: await locals.api.listModels() };
  } catch (cause) {
    if (!(cause instanceof ApiError)) throw cause;
    console.error(`${cause.name}: ${cause.message}`);
    catalog = { kind: 'unavailable' };
  }
  return { catalog, allowUploads: locals.config.allowUploads };
};
