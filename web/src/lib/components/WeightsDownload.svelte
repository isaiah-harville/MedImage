<script lang="ts">
import { Button } from '@sivir-ui/svelte/components/button';
import * as Card from '@sivir-ui/svelte/components/card';
import type { ModelSummary } from '#lib/contract.ts';
import { formatBytes } from '#lib/format.ts';

let { weights, license }: { weights: NonNullable<ModelSummary['weights']>; license: string } =
  $props();
</script>

<Card.Root>
  <Card.Header>
    <Card.Title><h2 id="weights-heading">Model weights</h2></Card.Title>
    <Card.Description>
      The Lightning checkpoint this page runs. The training data's license ({license}) also covers
      the weights.
    </Card.Description>
  </Card.Header>
  <Card.Content>
    <Button href={weights.url} variant="outline" download>
      Download checkpoint ({formatBytes(weights.bytes)})
    </Button>
    <p class="mt-3 font-mono text-[11px] break-all text-foreground-muted">SHA-256 {weights.sha256}</p>
  </Card.Content>
</Card.Root>
