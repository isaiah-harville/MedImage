import { z } from 'zod';

const labelMode = z.enum(['binary', 'multiclass', 'multilabel']);

const pngDataUrl = z
  .string()
  .regex(/^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/, 'expected a base64 PNG data URL');

export const modelSummarySchema = z
  .object({
    id: z.string().min(1),
    scan_type: z.string(),
    modality: z.string(),
    question: z.string(),
    label_mode: labelMode,
    classes: z.array(z.string()).min(1),
    input: z.object({ channels: z.number().int().positive(), size: z.number().int().positive() }),
    dataset: z.string(),
    license: z.string(),
    citation: z.string(),
    run_id: z.string(),
    test_auroc: z.object({ value: z.number(), lower: z.number(), upper: z.number() }).nullable(),
    demo_count: z.number().int().nonnegative(),
  })
  .transform((m) => ({
    id: m.id,
    scanType: m.scan_type,
    modality: m.modality,
    question: m.question,
    labelMode: m.label_mode,
    classes: m.classes,
    inputSize: m.input.size,
    dataset: m.dataset,
    license: m.license,
    citation: m.citation,
    runId: m.run_id,
    testAuroc: m.test_auroc,
    demoCount: m.demo_count,
  }));

export const demoSampleSchema = z
  .object({
    sample_id: z.string().min(1),
    truth: z.string(),
    image_url: z.string(),
  })
  .transform((s) => ({ sampleId: s.sample_id, truth: s.truth }));

export const predictionSchema = z
  .object({
    model: z.string(),
    label_mode: labelMode,
    probabilities: z
      .array(z.object({ class: z.string(), probability: z.number().min(0).max(1) }))
      .min(1),
    predicted: z.array(z.string()),
    threshold: z.number().nullable(),
    input_png: pngDataUrl,
    attribution_png: pngDataUrl,
    truth: z.string().nullable(),
    disclaimer: z.string(),
    latency_ms: z.number().nonnegative(),
  })
  .transform((p) => ({
    model: p.model,
    labelMode: p.label_mode,
    probabilities: p.probabilities
      .map((c) => ({ label: c.class, probability: c.probability }))
      .toSorted((a, b) => b.probability - a.probability),
    predicted: p.predicted,
    threshold: p.threshold,
    inputPng: p.input_png,
    attributionPng: p.attribution_png,
    truth: p.truth,
    disclaimer: p.disclaimer,
    latencyMs: p.latency_ms,
  }));

export type ModelSummary = z.output<typeof modelSummarySchema>;
export type DemoSample = z.output<typeof demoSampleSchema>;
export type Prediction = z.output<typeof predictionSchema>;
