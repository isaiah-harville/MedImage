import { describe, expect, it } from 'vitest';
import { groupByCategory } from '#lib/catalog.ts';
import type { ModelSummary } from '#lib/contract.ts';

const model = (id: string, category: string) => ({ id, category }) as ModelSummary;

describe('groupByCategory', () => {
  it('groups models by category in the order the API sends them', () => {
    const groups = groupByCategory([
      model('organamnist-convnext', 'CT and MRI'),
      model('chestmnist-convnext', 'X-ray and mammography'),
      model('cmmd-convnext', 'X-ray and mammography'),
      model('dermamnist-convnext', 'Skin'),
    ]);
    expect(groups.map((g) => [g.category, g.models.map((m) => m.id)])).toEqual([
      ['CT and MRI', ['organamnist-convnext']],
      ['X-ray and mammography', ['chestmnist-convnext', 'cmmd-convnext']],
      ['Skin', ['dermamnist-convnext']],
    ]);
  });
});
