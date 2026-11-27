<script lang="ts">
import type { Prediction } from '#lib/contract.ts';
import { formatLabel, formatPercent } from '#lib/format.ts';

let { prediction }: { prediction: Prediction } = $props();
</script>

<table class="w-full text-[13px]">
  <caption class="sr-only">Class probabilities, highest first</caption>
  <thead class="text-[12px] text-foreground-muted">
    <tr>
      <th scope="col" class="pb-2 text-left font-medium">Class</th>
      <th scope="col" class="w-2/5 pb-2 font-medium"><span class="sr-only">Bar</span></th>
      <th scope="col" class="pb-2 text-right font-medium">Probability</th>
    </tr>
  </thead>
  <tbody class="divide-y divide-border">
    {#each prediction.probabilities as item (item.label)}
      {@const chosen = prediction.predicted.includes(item.label)}
      <tr>
        <th scope="row" class={['py-2 text-left', chosen ? 'font-semibold' : 'font-normal text-foreground-muted']}>
          {formatLabel(item.label)}
          {#if chosen}<span class="sr-only">(predicted)</span>{/if}
        </th>
        <td class="px-3 py-2" aria-hidden="true">
          <div class="h-1.5 overflow-hidden rounded-full bg-secondary">
            <div
              class={['h-full rounded-full', chosen ? 'bg-primary' : 'bg-foreground-muted/40']}
              style:width={formatPercent(item.probability)}
            ></div>
          </div>
        </td>
        <td class="py-2 text-right font-mono tabular-nums">{formatPercent(item.probability)}</td>
      </tr>
    {/each}
  </tbody>
</table>
