import type { ModelSummary } from '#lib/contract.ts';

export type CategoryGroup = { category: string; models: ModelSummary[] };

export function groupByCategory(models: ModelSummary[]): CategoryGroup[] {
  const groups = new Map<string, ModelSummary[]>();
  for (const model of models) {
    groups.set(model.category, [...(groups.get(model.category) ?? []), model]);
  }
  return [...groups].map(([category, members]) => ({ category, models: members }));
}
