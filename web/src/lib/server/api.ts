import { z } from 'zod';
import {
  type DemoSample,
  demoSampleSchema,
  type ModelSummary,
  modelSummarySchema,
  type Prediction,
  predictionSchema,
} from '#lib/contract.ts';

export class ApiError extends Error {
  override name = 'ApiError';
  constructor(
    message: string,
    readonly status: number | null,
    options?: ErrorOptions,
  ) {
    super(message, options);
  }
}

export type ApiClient = ReturnType<typeof createApiClient>;

export function createApiClient({
  baseUrl,
  fetch,
}: {
  baseUrl: string;
  fetch: typeof globalThis.fetch;
}) {
  async function request(path: string, init?: RequestInit): Promise<Response> {
    let response: Response;
    try {
      response = await fetch(`${baseUrl}${path}`, { ...init, signal: AbortSignal.timeout(60_000) });
    } catch (cause) {
      throw new ApiError(`The inference service at ${baseUrl} did not answer ${path}.`, null, {
        cause,
      });
    }
    if (!response.ok) {
      await response.body?.cancel();
      throw new ApiError(
        `The inference service answered ${path} with ${response.status}.`,
        response.status,
      );
    }
    return response;
  }

  async function json<T extends z.ZodType>(schema: T, path: string, init?: RequestInit) {
    const response = await request(path, init);
    const result = schema.safeParse(await response.json());
    if (!result.success) {
      throw new ApiError(
        `The inference service answered ${path} with JSON that does not match deploy/api.md: ${z.prettifyError(result.error)}`,
        response.status,
      );
    }
    return result.data;
  }

  const model = (id: string) => `/models/${encodeURIComponent(id)}`;
  const sample = (id: string, sampleId: string) =>
    `${model(id)}/demo/${encodeURIComponent(sampleId)}`;

  return {
    listModels: (): Promise<ModelSummary[]> => json(z.array(modelSummarySchema), '/models'),

    listDemos: (id: string): Promise<DemoSample[]> =>
      json(z.array(demoSampleSchema), `${model(id)}/demo`),

    demoImage: (id: string, sampleId: string): Promise<Response> =>
      request(`${sample(id, sampleId)}.png`),

    predictDemo: (id: string, sampleId: string): Promise<Prediction> =>
      json(predictionSchema, '/predict', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: id, sample_id: sampleId }),
      }),

    predictUpload: async (id: string, file: File): Promise<Prediction> => {
      const image = Buffer.from(await file.arrayBuffer()).toString('base64');
      return json(predictionSchema, '/predict', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: id, image }),
      });
    },
  };
}
