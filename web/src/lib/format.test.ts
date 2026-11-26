import { describe, expect, it } from 'vitest';
import { verdict } from './format.ts';

describe('verdict', () => {
  it('is unknown for uploads, which have no ground truth', () => {
    expect(verdict({ labelMode: 'binary', predicted: ['pneumonia'], truth: null })).toBe('unknown');
  });

  it('matches a single class exactly, even one with a comma in its name', () => {
    const prediction = { labelMode: 'binary' as const, predicted: ['normal, benign'] };
    expect(verdict({ ...prediction, truth: 'normal, benign' })).toBe('correct');
    expect(verdict({ ...prediction, truth: 'malignant' })).toBe('wrong');
  });

  it('compares multilabel findings as sets', () => {
    const prediction = { labelMode: 'multilabel' as const, truth: 'mass, effusion' };
    expect(verdict({ ...prediction, predicted: ['effusion', 'mass'] })).toBe('correct');
    expect(verdict({ ...prediction, predicted: ['mass'] })).toBe('wrong');
  });
});
