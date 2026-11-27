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
  <p class="rounded-[var(--radius-lg)] bg-secondary px-4 py-6 text-foreground-muted">
    This model has no demo images yet. They appear here once its run publishes license-cleared
    test images.
  </p>
{:else}
  <form method="POST" action="?/demo" use:enhance={onsubmit}>
    <ul class="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {#each demos as demo, index (demo.sampleId)}
        {@const active = selected === demo.sampleId}
        {@const running = pending === demo.sampleId}
        <li>
          <button
            type="submit"
            name="sample_id"
            value={demo.sampleId}
            disabled={pending !== null}
            aria-current={active ? 'true' : undefined}
            class={[
              'group relative w-full overflow-hidden rounded-[var(--radius-lg)] bg-card p-1.5 text-left shadow-[var(--elevation-control)]',
              'transition-[box-shadow,scale] duration-150 ease-[var(--ease-press)] hover:shadow-[var(--elevation-float)] active:scale-[0.97] disabled:cursor-wait',
              active && 'shadow-[0_0_0_2px_var(--color-primary)]',
            ]}
          >
            <img
              src="/models/{encodeURIComponent(modelId)}/demo/{encodeURIComponent(demo.sampleId)}.png"
              alt="Demo image {index + 1}, labeled {formatLabel(demo.truth)} in the dataset"
              width="224"
              height="224"
              loading="lazy"
              class={[
                'aspect-square w-full rounded-[var(--radius-md)] bg-secondary object-cover transition-opacity',
                running && 'animate-pulse',
              ]}
            />
            <span class="mt-1.5 flex items-center justify-between gap-2 px-1 pb-0.5 text-[12px]">
              <span class="truncate text-foreground">{formatLabel(demo.truth)}</span>
              <span class="shrink-0 font-medium text-primary">{running ? 'Running' : 'Run'}</span>
            </span>
          </button>
        </li>
      {/each}
    </ul>
  </form>
{/if}
