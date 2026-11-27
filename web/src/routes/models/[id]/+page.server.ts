import { error } from '@sveltejs/kit';
import { ApiError } from '#lib/server/api.ts';
import { predictDemo, predictUpload } from '#lib/server/predict.ts';
import { uploadKind } from '#lib/upload.ts';
import type { Actions, PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ locals, params, parent }) => {
  const { catalog } = await parent();
  if (catalog.kind === 'unavailable') {
    error(503, 'The model service is not responding. Try again in a minute.');
  }
  const model = catalog.models.find((m) => m.id === params.id);
  if (!model) error(404, 'There is no model with that name.');
  try {
    return { model, demos: await locals.api.listDemos(model.id) };
  } catch (cause) {
    if (!(cause instanceof ApiError)) throw cause;
    console.error(`${cause.name}: ${cause.message}`);
    error(503, 'The model service is not responding. Try again in a minute.');
  }
};

export const actions = {
  demo: async ({ locals, params, request }) =>
    predictDemo({ api: locals.api, modelId: params.id, request }),
  upload: async ({ locals, params, request }) => {
    const models = await locals.api.listModels();
    const model = models.find((m) => m.id === params.id);
    if (!model) error(404, 'There is no model with that name.');
    return predictUpload({
      allowUploads: locals.config.allowUploads,
      api: locals.api,
      modelId: params.id,
      request,
      kind: uploadKind(model.inputDims),
    });
  },
} satisfies Actions;
