<script lang="ts">
import { Badge } from '@sivir-ui/svelte/components/badge';
import * as Card from '@sivir-ui/svelte/components/card';
import type { ModelSummary } from '#lib/contract.ts';
import AurocStat from './AurocStat.svelte';

let { model }: { model: ModelSummary } = $props();
</script>

<Card.Root
  class="group relative w-full transition-[box-shadow,translate] duration-200 ease-[var(--ease-out)] hover:-translate-y-0.5 hover:shadow-[var(--elevation-float)] has-[a:focus-visible]:shadow-[var(--focus-ring)] motion-reduce:hover:translate-y-0"
>
  <Card.Header>
    <div class="flex items-center justify-between gap-3">
      <Badge variant="secondary">{model.modality}</Badge>
      <span
        aria-hidden="true"
        class="text-foreground-muted transition-transform duration-200 ease-[var(--ease-out)] group-hover:translate-x-0.5"
        >→</span
      >
    </div>
    <Card.Title class="mt-3">
      <a href="/models/{model.id}" class="after:absolute after:inset-0 focus-visible:outline-none"
        >{model.scanType}</a
      >
    </Card.Title>
    <Card.Description>{model.question}</Card.Description>
  </Card.Header>
  <Card.Content class="mt-auto">
    <dl class="flex items-end justify-between gap-4 border-t border-border pt-4">
      <AurocStat auroc={model.testAuroc} />
      <div class="text-right">
        <dt class="sr-only">License</dt>
        <dd class="text-[12px] text-foreground-muted">{model.license}</dd>
      </div>
    </dl>
  </Card.Content>
</Card.Root>
