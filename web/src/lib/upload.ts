export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

export type UploadKind = 'image' | 'volume';

export const UPLOAD_FORMATS: Record<UploadKind, { accept: string; label: string }> = {
  image: { accept: 'image/png,image/jpeg', label: 'PNG or JPEG images' },
  volume: { accept: '.npy,.nii,.gz', label: '3D volumes as .npy or NIfTI (.nii, .nii.gz)' },
};

const IMAGE_TYPES = new Set(['image/png', 'image/jpeg']);
const VOLUME_NAME = /\.(npy|nii)(\.gz)?$/i;

export type UploadCheck = { ok: true } | { ok: false; status: 400 | 413 | 415; message: string };

export function uploadKind(dims: 2 | 3): UploadKind {
  return dims === 3 ? 'volume' : 'image';
}

export function checkUpload(
  file: { size: number; type: string; name: string },
  kind: UploadKind,
): UploadCheck {
  const formats = UPLOAD_FORMATS[kind].label;
  if (file.size === 0) {
    return { ok: false, status: 400, message: `Choose a file to upload: ${formats}.` };
  }
  const accepted = kind === 'volume' ? VOLUME_NAME.test(file.name) : IMAGE_TYPES.has(file.type);
  if (!accepted) {
    return { ok: false, status: 415, message: `This model reads ${formats}.` };
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return {
      ok: false,
      status: 413,
      message: `The file is ${formatMegabytes(file.size)}. The limit is 10 MB.`,
    };
  }
  return { ok: true };
}

function formatMegabytes(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
