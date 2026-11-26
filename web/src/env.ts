import { defineEnvVars } from '@sveltejs/kit/env';
import { allowUploadsSchema, apiUrlSchema } from './lib/server/config.ts';

export const variables = defineEnvVars({
  API_URL: {
    schema: apiUrlSchema,
    description: 'Base URL of the inference service. Only the SvelteKit server calls it.',
  },
  ALLOW_UPLOADS: {
    schema: allowUploadsSchema,
    description:
      'Uploads are on by default, as on diagnostics.harville.dev. Set to "false" only to turn them off.',
  },
});
