<script lang="ts">
import type { ModelSummary } from '#lib/contract.ts';
import AurocStat from './AurocStat.svelte';

let { model }: { model: ModelSummary } = $props();
</script>

<article
  class="relative flex w-full flex-col rounded-2xl bg-surface p-5 shadow-card transition-shadow duration-150 ease-out outline-accent outline-offset-2 hover:shadow-card-hover has-[a:focus-visible]:outline-2"
>
  <p class="text-sm text-fg-secondary">{model.modality}</p>
  <h3 class="mt-1 text-lg font-semibold tracking-tight">
    <a
      href="/models/{model.id}"
      class="after:absolute after:inset-0 after:rounded-2xl focus-visible:outline-none"
      >{model.scanType}</a
    >
  </h3>
  <p class="mt-1">{model.question}</p>
  <dl class="mt-auto space-y-3 pt-5">
    <AurocStat auroc={model.testAuroc} />
    <div>
      <dt class="sr-only">Dataset and license</dt>
      <dd class="text-sm text-fg-secondary">{model.dataset}, {model.license}</dd>
    </div>
  </dl>
</article>
