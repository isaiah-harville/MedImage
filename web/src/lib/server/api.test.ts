import { describe, expect, it } from 'vitest';
import {
  demosJson,
  modelsJson,
  PNG_DATA_URL_STUB,
  predictionJson,
} from '../../../tests/fixtures.ts';
import { ApiError, createApiClient } from './api.ts';

type Call = { url: string; method: string; body: RequestInit['body'] };

function fakeFetch(respond: (call: Call) => Response) {
  const calls: Call[] = [];
  const fetch: typeof globalThis.fetch = async (input, init) => {
    const call = { url: String(input), method: init?.method ?? 'GET', body: init?.body };
    calls.push(call);
    return respond(call);
  };
  return { fetch, calls };
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

const prediction = predictionJson({
  truth: 'pneumonia',
  inputPng: PNG_DATA_URL_STUB,
  attributionPng: PNG_DATA_URL_STUB,
});

describe('createApiClient', () => {
  it('maps GET /models to camelCase summaries, keeping a null AUROC', async () => {
    const { fetch, calls } = fakeFetch(() => json(modelsJson));
    const api = createApiClient({ baseUrl: 'http://api', fetch });

    const models = await api.listModels();

    expect(calls.map((c) => `${c.method} ${c.url}`)).toEqual(['GET http://api/models']);
    expect(models[0]).toEqual({
      id: 'pneumoniamnist-convnext',
      scanType: 'Pediatric chest X-ray',
      modality: 'X-ray',
      category: 'X-ray and mammography',
      question: 'Pneumonia or normal?',
      labelMode: 'binary',
      classes: ['normal', 'pneumonia'],
      inputSize: 224,
      dataset: 'PneumoniaMNIST (MedMNIST v2)',
      license: 'CC BY 4.0',
      citation: modelsJson[0]?.citation,
      runId: '20261127-031500-abcd1234',
      testAuroc: { value: 0.97, lower: 0.96, upper: 0.98 },
      demoCount: 4,
      weights: modelsJson[0]?.weights,
    });
    expect(models[1]?.testAuroc).toBeNull();
    expect(models[1]?.weights).toBeNull();
  });

  it('maps GET /models/{id}/demo and encodes the id in the path', async () => {
    const { fetch, calls } = fakeFetch(() => json(demosJson('pneumoniamnist-convnext')));
    const api = createApiClient({ baseUrl: 'http://api', fetch });

    const demos = await api.listDemos('pneumoniamnist-convnext');

    expect(calls[0]?.url).toBe('http://api/models/pneumoniamnist-convnext/demo');
    expect(demos[0]).toEqual({ sampleId: 'test-12', truth: 'pneumonia' });
    expect(demos).toHaveLength(4);
  });

  it('maps a demo Prediction and sorts probabilities highest first', async () => {
    const { fetch, calls } = fakeFetch(() => json(prediction));
    const api = createApiClient({ baseUrl: 'http://api', fetch });

    const result = await api.predictDemo('pneumoniamnist-convnext', 'test 12/../x');

    expect(calls[0]).toMatchObject({ method: 'POST', url: 'http://api/predict' });
    expect(JSON.parse(String(calls[0]?.body))).toEqual({
      model: 'pneumoniamnist-convnext',
      sample_id: 'test 12/../x',
    });
    expect(result).toEqual({
      model: 'pneumoniamnist-convnext',
      labelMode: 'binary',
      probabilities: [
        { label: 'pneumonia', probability: 0.92 },
        { label: 'normal', probability: 0.08 },
      ],
      predicted: ['pneumonia'],
      threshold: 0.41,
      inputPng: PNG_DATA_URL_STUB,
      attributionPng: PNG_DATA_URL_STUB,
      truth: 'pneumonia',
      disclaimer: 'Research model, not a medical device. Not for diagnosis or treatment.',
      latencyMs: 840,
    });
  });

  it('posts an upload as base64 in a JSON body', async () => {
    const { fetch, calls } = fakeFetch(() => json({ ...prediction, truth: null }));
    const api = createApiClient({ baseUrl: 'http://api', fetch });
    const file = new File([new Uint8Array([1, 2, 3])], 'scan.png', { type: 'image/png' });

    const result = await api.predictUpload('pneumoniamnist-convnext', file);

    expect(calls[0]?.url).toBe('http://api/predict');
    expect(JSON.parse(String(calls[0]?.body))).toEqual({
      model: 'pneumoniamnist-convnext',
      image: 'AQID',
    });
    expect(result.truth).toBeNull();
  });

  it('reports the HTTP status of an error response', async () => {
    const { fetch } = fakeFetch(() => json({ detail: 'too large' }, 413));
    const api = createApiClient({ baseUrl: 'http://api', fetch });
    const file = new File([new Uint8Array([1])], 'scan.png', { type: 'image/png' });

    const error = await api.predictUpload('pneumoniamnist-convnext', file).catch((e) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(413);
  });

  it('rejects a Prediction whose images are not PNG data URLs', async () => {
    const { fetch } = fakeFetch(() =>
      json({ ...prediction, attribution_png: 'https://evil.test/x.png' }),
    );
    const api = createApiClient({ baseUrl: 'http://api', fetch });

    await expect(api.predictDemo('pneumoniamnist-convnext', 'test-12')).rejects.toThrow(
      /attribution_png/,
    );
  });

  it('wraps a network failure with the service URL', async () => {
    const fetch: typeof globalThis.fetch = async () => {
      throw new TypeError('fetch failed');
    };
    const api = createApiClient({ baseUrl: 'http://api', fetch });

    await expect(api.listModels()).rejects.toMatchObject({
      name: 'ApiError',
      status: null,
      message: 'The inference service at http://api did not answer /models.',
    });
  });
});
