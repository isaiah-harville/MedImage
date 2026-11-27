import { defineEnvVars } from '@sveltejs/kit/env';
import { allowUploadsSchema, apiUrlSchema, papersDirSchema } from './lib/server/config.ts';

export const variables = defineEnvVars({
  API_URL: {
    schema: apiUrlSchema,
    description: 'Base URL of the inference service. Only the SvelteKit server calls it.',
  },
  PAPERS_DIR: {
    schema: papersDirSchema,
    description:
      'Folder of papers, one subfolder each with paper.yaml, the PDF, and preview.png. Defaults to ../papers, the repository folder, for local development.',
  },
  ALLOW_UPLOADS: {
    schema: allowUploadsSchema,
    description:
      'Uploads are on by default, as on diagnostics.harville.dev. Set to "false" only to turn them off.',
  },
});
