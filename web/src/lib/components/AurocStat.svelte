<script lang="ts">
import type { ModelSummary } from '#lib/contract.ts';
import { formatAuroc, formatInterval } from '#lib/format.ts';

let { auroc, large = false }: { auroc: ModelSummary['testAuroc']; large?: boolean } = $props();
</script>

<div>
  <dt class="text-[12px] font-medium tracking-wide text-foreground-muted uppercase">Test AUROC</dt>
  <dd
    class={[
      'font-mono font-semibold tabular-nums tracking-tight text-foreground',
      large ? 'text-[30px] leading-none' : 'text-[20px] leading-tight',
    ]}
  >
    {formatAuroc(auroc)}
  </dd>
  {#if auroc}
    <dd class="mt-1 font-mono text-[12px] tabular-nums text-foreground-muted">
      {formatInterval(auroc)}
    </dd>
  {/if}
</div>
