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
  disabled = false,
  onsubmit,
}: {
  modelId: string;
  demos: DemoSample[];
  selected: string | null;
  pending: string | null;
  disabled?: boolean;
  onsubmit: SubmitFunction;
} = $props();
</script>

{#if demos.length === 0}
  <p class="rounded-[var(--radius-lg)] bg-secondary px-4 py-6 text-[13px] text-foreground-muted">
    This model has no test cases yet. They appear here once its run publishes license-cleared
    test images.
  </p>
{:else}
  <form method="POST" action="?/demo" use:enhance={onsubmit}>
    <ul class="grid grid-cols-4 gap-2 lg:grid-cols-2">
      {#each demos as demo, index (demo.sampleId)}
        {@const active = selected === demo.sampleId}
        {@const running = pending === demo.sampleId}
        <li>
          <button
            type="submit"
            name="sample_id"
            value={demo.sampleId}
            disabled={disabled || pending !== null}
            aria-current={active ? 'true' : undefined}
            aria-label="Run case {index + 1}, reference label {formatLabel(demo.truth)}"
            class={[
              'w-full rounded-[var(--radius-lg)] p-1 text-left transition-[background-color,scale] duration-150 ease-[var(--ease-press)]',
              'hover:bg-secondary active:scale-[0.97] disabled:cursor-wait',
              active ? 'bg-accent-tint outline-2 outline-primary' : '',
            ]}
          >
            <img
              src="/models/{encodeURIComponent(modelId)}/demo/{encodeURIComponent(demo.sampleId)}.png"
              alt=""
              width="224"
              height="224"
              loading="lazy"
              class={['aspect-square w-full rounded-[var(--radius-md)] bg-black object-cover', running && 'animate-pulse']}
            />
            <span class="mt-1 hidden px-0.5 text-[11px] leading-snug sm:block">
              <span class="font-mono text-foreground-muted">Case {String(index + 1).padStart(2, '0')}</span>
              <span class="block truncate">{running ? 'Running' : formatLabel(demo.truth)}</span>
            </span>
          </button>
        </li>
      {/each}
    </ul>
  </form>
{/if}
