import { isHttpError } from '@sveltejs/kit';
import { describe, expect, it } from 'vitest';
import { PNG_DATA_URL_STUB, predictionJson } from '../../../tests/fixtures.ts';
import { createApiClient } from './api.ts';
import { CONSENT_REQUIRED, predictUpload } from './predict.ts';

function uploadRequest(file: File, { consent = true } = {}): Request {
  const body = new FormData();
  body.set('file', file);
  if (consent) body.set('consent', 'yes');
  return new Request('http://web/models/pneumoniamnist-convnext?/upload', { method: 'POST', body });
}

const png = () => new File([new Uint8Array([137, 80, 78, 71])], 'scan.png', { type: 'image/png' });

function apiReturning(status: number, body: unknown) {
  const calls: string[] = [];
  const api = createApiClient({
    baseUrl: 'http://api',
    fetch: async (input) => {
      calls.push(String(input));
      return new Response(JSON.stringify(body), { status });
    },
  });
  return { api, calls };
}

describe('upload action', () => {
  it('returns 404 without calling the API when uploads are disabled', async () => {
    const { api, calls } = apiReturning(200, {});

    const thrown = await predictUpload({
      allowUploads: false,
      api,
      modelId: 'pneumoniamnist-convnext',
      request: uploadRequest(png()),
    }).catch((e: unknown) => e);

    expect(isHttpError(thrown, 404)).toBe(true);
    expect(calls).toEqual([]);
  });

  it('returns 400 without calling the API when consent is missing', async () => {
    const { api, calls } = apiReturning(200, {});

    const result = await predictUpload({
      allowUploads: true,
      api,
      modelId: 'pneumoniamnist-convnext',
      request: uploadRequest(png(), { consent: false }),
    });

    expect(result).toMatchObject({
      status: 400,
      data: { source: 'upload', message: CONSENT_REQUIRED },
    });
    expect(calls).toEqual([]);
  });

  it('returns the prediction when uploads are enabled', async () => {
    const { api, calls } = apiReturning(
      200,
      predictionJson({
        truth: null,
        inputPng: PNG_DATA_URL_STUB,
        attributionPng: PNG_DATA_URL_STUB,
      }),
    );

    const result = await predictUpload({
      allowUploads: true,
      api,
      modelId: 'pneumoniamnist-convnext',
      request: uploadRequest(png()),
    });

    expect(calls).toEqual(['http://api/models/pneumoniamnist-convnext/predict']);
    expect(result).toMatchObject({ source: 'upload', sampleId: null, prediction: { truth: null } });
  });

  it('rejects a GIF with 415 before calling the API', async () => {
    const { api, calls } = apiReturning(200, {});
    const gif = new File([new Uint8Array([71, 73, 70])], 'scan.gif', { type: 'image/gif' });

    const result = await predictUpload({
      allowUploads: true,
      api,
      modelId: 'pneumoniamnist-convnext',
      request: uploadRequest(gif),
    });

    expect(result).toMatchObject({ status: 415, data: { source: 'upload' } });
    expect(calls).toEqual([]);
  });

  it('turns an API 422 into a message the user can act on', async () => {
    const { api } = apiReturning(422, { detail: 'cannot decode' });

    const result = await predictUpload({
      allowUploads: true,
      api,
      modelId: 'pneumoniamnist-convnext',
      request: uploadRequest(png()),
    });

    expect(result).toMatchObject({
      status: 422,
      data: {
        message: 'The file could not be read as an image. Check that it opens in an image viewer.',
      },
    });
  });
});
