import type { Handle } from '@sveltejs/kit/hooks';
import { createApiClient } from '#lib/server/api.ts';
import type { Config } from '#lib/server/config.ts';
import { withPrivacyHeaders } from '#lib/server/headers.ts';
import { ALLOW_UPLOADS, API_URL } from '$app/env/private';

const config: Config = { apiUrl: API_URL, allowUploads: ALLOW_UPLOADS };
const api = createApiClient({ baseUrl: config.apiUrl, fetch });

export const handle: Handle = async ({ event, resolve }) => {
  event.locals.config = config;
  event.locals.api = api;
  return withPrivacyHeaders(event.request.method, await resolve(event));
};
