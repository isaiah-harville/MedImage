import { error, fail } from '@sveltejs/kit';
import { checkUpload, type UploadKind } from '#lib/upload.ts';
import { type ApiClient, ApiError } from './api.ts';

type Deps = { api: ApiClient; modelId: string; request: Request };

export type PredictFailure = { source: 'demo' | 'upload'; message: string };

export async function predictDemo({ api, modelId, request }: Deps) {
  const form = await request.formData();
  const sampleId = form.get('sample_id');
  if (typeof sampleId !== 'string' || sampleId === '') {
    return fail(400, { source: 'demo', message: 'Pick a demo image.' } satisfies PredictFailure);
  }
  try {
    return {
      source: 'demo' as const,
      sampleId,
      prediction: await api.predictDemo(modelId, sampleId),
    };
  } catch (cause) {
    return failure('demo', cause);
  }
}

export async function predictUpload({
  allowUploads,
  api,
  modelId,
  request,
  kind,
}: Deps & { allowUploads: boolean; kind: UploadKind }) {
  if (!allowUploads) error(404, 'Uploads are not available on this site.');
  const form = await request.formData();
  if (form.get('consent') !== 'yes') {
    return fail(400, {
      source: 'upload',
      message: CONSENT_REQUIRED,
    } satisfies PredictFailure);
  }
  const file = form.get('file');
  if (!(file instanceof File)) {
    return fail(400, {
      source: 'upload',
      message: 'Choose a file to upload.',
    } satisfies PredictFailure);
  }
  const check = checkUpload(file, kind);
  if (!check.ok) {
    return fail(check.status, {
      source: 'upload',
      message: check.message,
    } satisfies PredictFailure);
  }
  try {
    return {
      source: 'upload' as const,
      sampleId: null,
      prediction: await api.predictUpload(modelId, file),
    };
  } catch (cause) {
    return failure('upload', cause, kind);
  }
}

export const CONSENT_REQUIRED =
  'Confirm that the image has no identifiers, that you have the right to share it, and that the result is not medical advice.';

const UPLOAD_ERRORS = {
  image: {
    413: 'The image is too large. The limit is 10 MB and 4096 x 4096 pixels.',
    415: 'Only PNG and JPEG images are accepted.',
    422: 'The file could not be read as an image. Check that it opens in an image viewer.',
  },
  volume: {
    413: 'The volume is too large. The limit is 10 MB.',
    415: 'Only .npy arrays and NIfTI (.nii, .nii.gz) volumes are accepted.',
    422: 'The file could not be read as a 3D volume. Upload a .npy array or a NIfTI (.nii, .nii.gz) file.',
  },
} as const;

function failure(source: PredictFailure['source'], cause: unknown, kind: UploadKind = 'image') {
  if (!(cause instanceof ApiError)) throw cause;
  switch (cause.status) {
    case 404:
      error(404, 'That model or demo image does not exist.');
      break;
    case 413:
    case 415:
    case 422:
      return fail(cause.status, {
        source,
        message: UPLOAD_ERRORS[kind][cause.status],
      } satisfies PredictFailure);
  }
  console.error(`${cause.name}: ${cause.message}`);
  return fail(502, {
    source,
    message: 'The model service is not responding. Try again in a minute.',
  } satisfies PredictFailure);
}
