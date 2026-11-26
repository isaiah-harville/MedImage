<script lang="ts">
import type { Prediction } from '#lib/contract.ts';
import { formatLabel, formatPercent } from '#lib/format.ts';

let { prediction }: { prediction: Prediction } = $props();
</script>

<ul class="space-y-3" aria-label="Class probabilities, highest first">
  {#each prediction.probabilities as item (item.label)}
    {@const chosen = prediction.predicted.includes(item.label)}
    <li>
      <div class="flex items-baseline justify-between gap-3 text-sm">
        <span class={chosen ? 'font-semibold text-fg' : 'text-fg-secondary'}>
          {formatLabel(item.label)}
          {#if chosen}<span class="sr-only">(predicted)</span>{/if}
        </span>
        <span class="font-mono tabular-nums">{formatPercent(item.probability)}</span>
      </div>
      <div class="mt-1 h-2 overflow-hidden rounded-full bg-muted" aria-hidden="true">
        <div
          class={['h-full rounded-full', chosen ? 'bg-accent' : 'bg-fg-secondary/50']}
          style:width={formatPercent(item.probability)}
        ></div>
      </div>
    </li>
  {/each}
</ul>
