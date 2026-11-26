import { crc32, deflateSync } from 'node:zlib';

function chunk(type: string, data: Buffer): Buffer {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([length, body, crc]);
}

export function syntheticPng({
  size,
  seed,
  heat,
}: {
  size: number;
  seed: number;
  heat: boolean;
}): Buffer {
  const channels = 3;
  const rows: Buffer[] = [];
  const cx = size * (0.35 + (0.3 * ((seed * 7) % 10)) / 10);
  const cy = size * 0.45;
  for (let y = 0; y < size; y++) {
    const row = Buffer.alloc(1 + size * channels);
    for (let x = 0; x < size; x++) {
      const ribs = 0.5 + 0.5 * Math.sin((y + seed * 3) / 6);
      const lung = Math.max(
        0,
        1 - Math.hypot((x - size / 2) / (size * 0.45), (y - size / 2) / (size * 0.5)),
      );
      const gray = Math.round(30 + 160 * lung * (0.6 + 0.4 * ribs));
      let [r, g, b] = [gray, gray, gray];
      if (heat) {
        const h = Math.max(0, 1 - Math.hypot(x - cx, y - cy) / (size * 0.22));
        r = Math.round(gray * (1 - h) + 255 * h);
        g = Math.round(gray * (1 - h) + 120 * h);
        b = Math.round(gray * (1 - h) + 20 * h);
      }
      row.set([r, g, b], 1 + x * channels);
    }
    rows.push(row);
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(size, 0);
  header.writeUInt32BE(size, 4);
  header.set([8, 2, 0, 0, 0], 8);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(Buffer.concat(rows))),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}
