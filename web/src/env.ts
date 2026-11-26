import { defineEnvVars } from '@sveltejs/kit/env';
import { allowUploadsSchema, apiUrlSchema } from './lib/server/config.ts';

export const variables = defineEnvVars({
  API_URL: {
    schema: apiUrlSchema,
    description: 'Base URL of the inference service. Only the SvelteKit server calls it.',
  },
  ALLOW_UPLOADS: {
    schema: allowUploadsSchema,
    description: 'Set to "true" on the internal site to enable image uploads. Defaults to false.',
  },
});
