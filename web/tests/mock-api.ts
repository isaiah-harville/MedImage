import { createServer } from 'node:http';
import { demosJson, modelsJson, predictionJson } from './fixtures.ts';
import { syntheticPng } from './png.ts';

const port = Number(process.env.MOCK_API_PORT ?? 8787);
const dataUrl = (png: Buffer) => `data:image/png;base64,${png.toString('base64')}`;

function seedOf(sampleId: string): number {
  return [...sampleId].reduce((sum, ch) => sum + ch.charCodeAt(0), 0);
}

function prediction(model: string, truth: string | null, seed: number) {
  const volumetric = modelsJson.find((m) => m.id === model)?.input.dims === 3;
  const slice = (depth: number, heat: boolean) =>
    dataUrl(syntheticPng({ size: 64, seed: seed + depth, heat }));
  const depths = [...Array(volumetric ? 64 : 0).keys()];
  return predictionJson({
    truth,
    inputPng: dataUrl(syntheticPng({ size: 224, seed, heat: false })),
    attributionPng: dataUrl(syntheticPng({ size: 224, seed, heat: true })),
    slices: volumetric
      ? {
          input: depths.map((d) => slice(d, false)),
          attribution: depths.map((d) => slice(d, true)),
        }
      : null,
  });
}

const server = createServer((req, res) => {
  const url = new URL(req.url ?? '/', 'http://mock');
  const parts = url.pathname.split('/').filter(Boolean).map(decodeURIComponent);
  const json = (status: number, body: unknown) => {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(body));
  };
  if (req.method === 'GET' && url.pathname === '/health') return res.end('ok');
  if (req.method === 'GET' && url.pathname === '/models') return json(200, modelsJson);
  if (req.method === 'POST' && url.pathname === '/predict') {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => {
      const body = JSON.parse(Buffer.concat(chunks).toString());
      if (!modelsJson.some((m) => m.id === body.model)) return json(404, { detail: 'no model' });
      if (body.image) return json(200, prediction(body.model, null, 1));
      const demo = demosJson(body.model).find((d) => d.sample_id === body.sample_id);
      if (!demo) return json(404, { detail: 'no demo image' });
      setTimeout(() => json(200, prediction(body.model, demo.truth, seedOf(demo.sample_id))), 300);
    });
    return;
  }
  const model = modelsJson.find((m) => m.id === parts[1]);
  if (parts[0] !== 'models' || !model) return json(404, { detail: 'not found' });

  const demos = demosJson(model.id);
  if (req.method === 'GET' && parts.length === 3 && parts[2] === 'demo') return json(200, demos);
  if (req.method === 'GET' && parts.length === 4 && parts[3]?.endsWith('.png')) {
    const sampleId = parts[3].slice(0, -'.png'.length);
    if (!demos.some((d) => d.sample_id === sampleId)) return json(404, { detail: 'not found' });
    res.writeHead(200, { 'Content-Type': 'image/png' });
    return res.end(syntheticPng({ size: 224, seed: seedOf(sampleId), heat: false }));
  }
  return json(404, { detail: 'not found' });
});

server.listen(port, () => {
  console.log(`mock inference API on http://localhost:${port}`);
});
