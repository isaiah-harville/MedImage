import { createServer } from 'node:http';
import { demosJson, modelsJson, predictionJson } from './fixtures.ts';
import { syntheticPng } from './png.ts';

const port = Number(process.env.MOCK_API_PORT ?? 8787);
const dataUrl = (png: Buffer) => `data:image/png;base64,${png.toString('base64')}`;

function seedOf(sampleId: string): number {
  return [...sampleId].reduce((sum, ch) => sum + ch.charCodeAt(0), 0);
}

function prediction(truth: string | null, seed: number) {
  return predictionJson({
    truth,
    inputPng: dataUrl(syntheticPng({ size: 224, seed, heat: false })),
    attributionPng: dataUrl(syntheticPng({ size: 224, seed, heat: true })),
  });
}

const server = createServer((req, res) => {
  const url = new URL(req.url ?? '/', 'http://mock');
  const parts = url.pathname.split('/').filter(Boolean).map(decodeURIComponent);
  const json = (status: number, body: unknown) => {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(body));
  };
  const model = modelsJson.find((m) => m.id === parts[1]);

  if (req.method === 'GET' && url.pathname === '/healthz') {
    return json(200, { status: 'ok', models: modelsJson.length });
  }
  if (req.method === 'GET' && url.pathname === '/models') return json(200, modelsJson);
  if (parts[0] !== 'models' || !model) return json(404, { detail: 'not found' });

  const demos = demosJson(model.id);
  if (req.method === 'GET' && parts.length === 3 && parts[2] === 'demo') return json(200, demos);
  if (req.method === 'GET' && parts.length === 4 && parts[3]?.endsWith('.png')) {
    const sampleId = parts[3].slice(0, -'.png'.length);
    if (!demos.some((d) => d.sample_id === sampleId)) return json(404, { detail: 'not found' });
    res.writeHead(200, { 'Content-Type': 'image/png' });
    return res.end(syntheticPng({ size: 224, seed: seedOf(sampleId), heat: false }));
  }
  if (req.method === 'POST' && parts.length === 5 && parts[4] === 'predict') {
    const demo = demos.find((d) => d.sample_id === parts[3]);
    if (!demo) return json(404, { detail: 'not found' });
    return setTimeout(() => json(200, prediction(demo.truth, seedOf(demo.sample_id))), 300);
  }
  if (req.method === 'POST' && parts.length === 3 && parts[2] === 'predict') {
    req.resume();
    req.on('end', () => json(200, prediction(null, 1)));
    return;
  }
  return json(404, { detail: 'not found' });
});

server.listen(port, () => {
  console.log(`mock inference API on http://localhost:${port}`);
});
