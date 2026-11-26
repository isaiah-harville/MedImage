import type { ModelSummary, Prediction } from '#lib/contract.ts';

export function formatPercent(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`;
}

export function formatAuroc(auroc: ModelSummary['testAuroc']): string {
  if (auroc === null) return 'Not measured';
  return auroc.value.toFixed(3);
}

export function formatInterval(auroc: NonNullable<ModelSummary['testAuroc']>): string {
  return `95% CI ${auroc.lower.toFixed(3)} to ${auroc.upper.toFixed(3)}`;
}

export function formatLatency(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

export function formatLabel(label: string): string {
  const spaced = label.replaceAll('_', ' ');
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

export type Verdict = 'correct' | 'wrong' | 'unknown';

export function verdict(
  prediction: Pick<Prediction, 'labelMode' | 'predicted' | 'truth'>,
): Verdict {
  if (prediction.truth === null) return 'unknown';
  const { labelMode, predicted, truth } = prediction;
  if (labelMode !== 'multilabel') {
    return predicted.length === 1 && predicted[0] === truth ? 'correct' : 'wrong';
  }
  // The API joins a multilabel truth with ", "; ChestMNIST's labels contain no commas.
  const expected = new Set(truth.split(', ').filter(Boolean));
  const same = predicted.length === expected.size && predicted.every((p) => expected.has(p));
  return same ? 'correct' : 'wrong';
}

export type DatasetCredit = Pick<ModelSummary, 'dataset' | 'license' | 'citation'>;

export function datasetCredits(models: ModelSummary[]): DatasetCredit[] {
  const byDataset = new Map<string, DatasetCredit>();
  for (const { dataset, license, citation } of models) {
    if (!byDataset.has(dataset)) byDataset.set(dataset, { dataset, license, citation });
  }
  return [...byDataset.values()];
}
