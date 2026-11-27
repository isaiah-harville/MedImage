import { describe, expect, it } from 'vitest';
import { checkUpload, MAX_UPLOAD_BYTES } from './upload.ts';

const png = (size: number, type = 'image/png') => ({ size, type, name: 'scan.png' });
const volume = (name: string, size = 10) => ({ size, type: '', name });

describe('checkUpload', () => {
  it('accepts PNG and JPEG up to 10 MB for an image model', () => {
    expect(checkUpload(png(MAX_UPLOAD_BYTES), 'image')).toEqual({ ok: true });
    expect(checkUpload(png(1, 'image/jpeg'), 'image')).toEqual({ ok: true });
  });

  it('rejects a file one byte over 10 MB with 413', () => {
    expect(checkUpload(png(MAX_UPLOAD_BYTES + 1), 'image')).toMatchObject({
      ok: false,
      status: 413,
    });
  });

  it('rejects other types with 415 and an empty file with 400', () => {
    expect(checkUpload(png(10, 'image/webp'), 'image')).toMatchObject({ ok: false, status: 415 });
    expect(checkUpload(png(0), 'image')).toMatchObject({ ok: false, status: 400 });
  });

  it('accepts .npy and NIfTI volumes by name for a 3D model, and refuses images', () => {
    for (const name of ['ct.npy', 'ct.NII', 'ct.nii.gz', 'ct.npy.gz']) {
      expect(checkUpload(volume(name), 'volume')).toEqual({ ok: true });
    }
    expect(checkUpload(png(10), 'volume')).toMatchObject({ ok: false, status: 415 });
    expect(checkUpload(volume('ct.dcm'), 'volume')).toMatchObject({ ok: false, status: 415 });
  });
});
