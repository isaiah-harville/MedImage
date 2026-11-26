import { describe, expect, it } from 'vitest';
import { allowUploadsSchema, apiUrlSchema } from './config.ts';

function messages(result: { success: boolean; error?: { issues: { message: string }[] } }) {
  return result.error?.issues.map((issue) => issue.message) ?? [];
}

describe('API_URL', () => {
  it('accepts an http URL and drops trailing slashes', () => {
    expect(apiUrlSchema.parse('http://inference:8000/')).toBe('http://inference:8000');
  });

  it('accepts an https URL with a path prefix', () => {
    expect(apiUrlSchema.parse('https://api.example.org/demo')).toBe('https://api.example.org/demo');
  });

  it('is required and says how to set it', () => {
    expect(messages(apiUrlSchema.safeParse(undefined))).toEqual([
      'API_URL is required. Set it to the inference service base URL, for example API_URL=http://localhost:8000.',
    ]);
  });

  it('rejects a value that is not an http(s) URL', () => {
    for (const value of ['localhost:8000', 'ftp://inference', 'not a url']) {
      expect(messages(apiUrlSchema.safeParse(value))).toEqual([
        'API_URL must be an http(s) URL, for example http://localhost:8000.',
      ]);
    }
  });
});

describe('ALLOW_UPLOADS', () => {
  it('defaults to true when unset or empty', () => {
    expect(allowUploadsSchema.parse(undefined)).toBe(true);
    expect(allowUploadsSchema.parse('')).toBe(true);
  });

  it('reads "true" and "false"', () => {
    expect(allowUploadsSchema.parse('true')).toBe(true);
    expect(allowUploadsSchema.parse('false')).toBe(false);
  });

  it('rejects anything else instead of guessing', () => {
    for (const value of ['1', 'yes', 'TRUE']) {
      expect(messages(allowUploadsSchema.safeParse(value))).toEqual([
        'ALLOW_UPLOADS must be "true" or "false". Leave it unset to allow uploads.',
      ]);
    }
  });
});
