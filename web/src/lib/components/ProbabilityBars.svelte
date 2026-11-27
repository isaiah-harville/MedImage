<script lang="ts">
import type { Prediction } from '#lib/contract.ts';
import { formatLabel, formatPercent } from '#lib/format.ts';

let { prediction }: { prediction: Prediction } = $props();
</script>

<ul class="space-y-3" aria-label="Class probabilities, highest first">
  {#each prediction.probabilities as item (item.label)}
    {@const chosen = prediction.predicted.includes(item.label)}
    <li>
      <div class="flex items-baseline justify-between gap-3 text-[13px]">
        <span class={chosen ? 'font-semibold text-foreground' : 'text-foreground-muted'}>
          {formatLabel(item.label)}
          {#if chosen}<span class="sr-only">(predicted)</span>{/if}
        </span>
        <span class="font-mono tabular-nums">{formatPercent(item.probability)}</span>
      </div>
      <div class="mt-1.5 h-1.5 overflow-hidden rounded-full bg-secondary" aria-hidden="true">
        <div
          class={[
            'bar h-full origin-left rounded-full',
            chosen ? 'bg-primary' : 'bg-foreground-muted/40',
          ]}
          style:width={formatPercent(item.probability)}
        ></div>
      </div>
    </li>
  {/each}
</ul>

<style>
  @keyframes grow {
    from {
      transform: scaleX(0);
    }
  }

  .bar {
    animation: grow 480ms var(--ease-out) both;
  }
</style>
