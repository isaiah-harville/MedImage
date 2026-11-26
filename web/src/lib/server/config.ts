import { z } from 'zod';

export const apiUrlSchema = z
  .url({
    protocol: /^https?$/,
    error: (issue) =>
      issue.input === undefined || issue.input === ''
        ? 'API_URL is required. Set it to the inference service base URL, for example API_URL=http://localhost:8000.'
        : 'API_URL must be an http(s) URL, for example http://localhost:8000.',
  })
  .transform((url) => url.replace(/\/+$/, ''));

export const allowUploadsSchema = z
  .enum(['true', 'false', ''], {
    error: 'ALLOW_UPLOADS must be "true" or "false". Leave it unset to allow uploads.',
  })
  .optional()
  .transform((value) => value !== 'false');

export type Config = {
  apiUrl: string;
  allowUploads: boolean;
};
