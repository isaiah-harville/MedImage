<script lang="ts">
import type { DemoSample } from '#lib/contract.ts';
import { formatLabel } from '#lib/format.ts';
import type { SubmitFunction } from '$app/forms';
import { enhance } from '$app/forms';

let {
  modelId,
  demos,
  selected,
  pending,
  onsubmit,
}: {
  modelId: string;
  demos: DemoSample[];
  selected: string | null;
  pending: string | null;
  onsubmit: SubmitFunction;
} = $props();
</script>

{#if demos.length === 0}
  <p class="rounded-xl bg-muted px-4 py-6 text-fg-secondary">
    This model has no demo images yet. They appear here once the model service publishes curated
    test images.
  </p>
{:else}
  <form method="POST" action="?/demo" use:enhance={onsubmit}>
    <ul class="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {#each demos as demo, index (demo.sampleId)}
        <li>
          <button
            type="submit"
            name="sample_id"
            value={demo.sampleId}
            disabled={pending !== null}
            aria-current={selected === demo.sampleId ? 'true' : undefined}
            class={[
              'group w-full rounded-xl bg-surface p-2 text-left shadow-btn transition duration-150 ease-out',
              'hover:shadow-card-hover active:scale-[0.97] disabled:cursor-wait',
              selected === demo.sampleId && 'ring-2 ring-accent',
            ]}
          >
            <img
              src="/models/{encodeURIComponent(modelId)}/demo/{encodeURIComponent(demo.sampleId)}.png"
              alt="Demo image {index + 1}, labeled {formatLabel(demo.truth)} in the dataset"
              width="224"
              height="224"
              loading="lazy"
              class="aspect-square w-full rounded-lg bg-muted object-cover"
            />
            <span class="mt-2 flex items-center justify-between gap-2 px-1 text-sm">
              <span class="text-fg-secondary">Truth: <span class="text-fg">{formatLabel(demo.truth)}</span></span>
              <span class="font-medium text-accent">
                {pending === demo.sampleId ? 'Running' : 'Run'}
              </span>
            </span>
          </button>
        </li>
      {/each}
    </ul>
  </form>
{/if}
