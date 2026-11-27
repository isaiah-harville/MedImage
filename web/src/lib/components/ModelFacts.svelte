<script lang="ts">
import type { ModelSummary } from '#lib/contract.ts';
import { formatAuroc, formatLabel } from '#lib/format.ts';

let { model }: { model: ModelSummary } = $props();
</script>

<dl
  class="grid grid-cols-2 gap-x-6 gap-y-4 rounded-[var(--radius-xl)] bg-card px-5 py-4 text-[13px] shadow-[var(--elevation-control)] md:grid-cols-[repeat(4,minmax(0,1fr))_minmax(0,2fr)]"
>
  <div>
    <dt class="text-[12px] text-foreground-muted">Intended use</dt>
    <dd class="mt-0.5 font-medium">Research only</dd>
  </div>
  <div>
    <dt class="text-[12px] text-foreground-muted">Test AUROC</dt>
    <dd class="mt-0.5 font-mono font-medium tabular-nums">{formatAuroc(model.testAuroc)}</dd>
    {#if model.testAuroc}
      <dd class="font-mono text-[11px] text-foreground-muted tabular-nums">
        95% CI {model.testAuroc.lower.toFixed(3)} to {model.testAuroc.upper.toFixed(3)}
      </dd>
    {/if}
  </div>
  <div>
    <dt class="text-[12px] text-foreground-muted">Training data</dt>
    <dd class="mt-0.5 font-medium">{model.dataset}</dd>
  </div>
  <div>
    <dt class="text-[12px] text-foreground-muted">License</dt>
    <dd class="mt-0.5 font-medium">{model.license}</dd>
  </div>
  <div class="col-span-2 md:col-span-1">
    <dt class="text-[12px] text-foreground-muted">Output classes</dt>
    <dd class="mt-0.5">{model.classes.map(formatLabel).join(', ')}</dd>
  </div>
</dl>
