<script lang="ts">
import { Button } from '@sivir-ui/svelte/components/button';
import type { ModelSummary } from '#lib/contract.ts';
import { formatBytes } from '#lib/format.ts';

let { weights, license }: { weights: NonNullable<ModelSummary['weights']>; license: string } =
  $props();
</script>

<section aria-labelledby="weights-heading" class="rounded-[var(--radius-xl)] bg-card p-4 shadow-[var(--elevation-control)]">
  <h2 id="weights-heading" class="text-[14px] font-semibold">Model weights</h2>
  <p class="mt-0.5 text-[12px] text-foreground-muted">
    The Lightning checkpoint this page runs, under the training data's license ({license}).
  </p>
  <Button href={weights.url} variant="outline" size="sm" download class="mt-3">
    Download checkpoint, {formatBytes(weights.bytes)}
  </Button>
  <p class="mt-2 font-mono text-[10.5px] break-all text-foreground-muted">SHA-256 {weights.sha256}</p>
</section>
