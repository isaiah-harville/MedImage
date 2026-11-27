<script lang="ts">
import type { ModelSummary } from '#lib/contract.ts';
import { formatBytes } from '#lib/format.ts';

let { weights, license }: { weights: NonNullable<ModelSummary['weights']>; license: string } =
  $props();
</script>

<section aria-labelledby="weights-heading" class="mt-10 max-w-xl">
  <h2 id="weights-heading" class="text-xl font-semibold tracking-tight">Model weights</h2>
  <p class="mt-1 text-sm text-fg-secondary">
    The Lightning checkpoint this page runs. Load it with <code>opendiagnostics</code>; the
    training data's license ({license}) also covers the weights.
  </p>
  <a
    href={weights.url}
    download
    class="mt-4 inline-flex h-10 items-center rounded-lg bg-accent px-4 text-sm font-medium text-accent-fg transition duration-150 ease-out hover:bg-accent-hover active:scale-[0.97]"
  >
    Download weights ({formatBytes(weights.bytes)})
  </a>
  <p class="mt-3 break-all font-mono text-xs text-fg-secondary">SHA-256 {weights.sha256}</p>
</section>
