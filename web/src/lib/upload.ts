export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
export const ACCEPTED_UPLOAD_TYPES = ['image/png', 'image/jpeg'] as const;

export type UploadCheck = { ok: true } | { ok: false; status: 400 | 413 | 415; message: string };

export function checkUpload(file: { size: number; type: string }): UploadCheck {
  if (file.size === 0) {
    return { ok: false, status: 400, message: 'Choose a PNG or JPEG image to upload.' };
  }
  if (!ACCEPTED_UPLOAD_TYPES.some((type) => type === file.type)) {
    return {
      ok: false,
      status: 415,
      message: 'Only PNG and JPEG images are accepted. Export the image as PNG or JPEG and retry.',
    };
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return {
      ok: false,
      status: 413,
      message: `The image is ${formatMegabytes(file.size)}. The limit is 10 MB.`,
    };
  }
  return { ok: true };
}

function formatMegabytes(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
