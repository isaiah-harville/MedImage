import { describe, expect, it } from 'vitest';
import { checkUpload, MAX_UPLOAD_BYTES } from './upload.ts';

describe('checkUpload', () => {
  it('accepts PNG and JPEG up to 10 MB', () => {
    expect(checkUpload({ type: 'image/png', size: MAX_UPLOAD_BYTES })).toEqual({ ok: true });
    expect(checkUpload({ type: 'image/jpeg', size: 1 })).toEqual({ ok: true });
  });

  it('rejects a file one byte over 10 MB with 413', () => {
    expect(checkUpload({ type: 'image/png', size: MAX_UPLOAD_BYTES + 1 })).toMatchObject({
      ok: false,
      status: 413,
    });
  });

  it('rejects other types with 415 and an empty file with 400', () => {
    expect(checkUpload({ type: 'image/webp', size: 10 })).toMatchObject({ ok: false, status: 415 });
    expect(checkUpload({ type: 'image/png', size: 0 })).toMatchObject({ ok: false, status: 400 });
  });
});
