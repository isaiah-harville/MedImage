<script lang="ts">
import type { ModelSummary } from '#lib/contract.ts';
import { formatAuroc, formatInterval } from '#lib/format.ts';

let { auroc }: { auroc: ModelSummary['testAuroc'] } = $props();
</script>

<div>
  <dt class="text-xs text-fg-secondary">Test AUROC</dt>
  <dd class="font-mono text-lg font-semibold tabular-nums text-fg">{formatAuroc(auroc)}</dd>
  {#if auroc}
    <dd class="font-mono text-xs tabular-nums text-fg-secondary">{formatInterval(auroc)}</dd>
  {/if}
</div>
