<script lang="ts">
import type { Verdict } from '#lib/format.ts';

let { verdict, truth }: { verdict: Exclude<Verdict, 'unknown'>; truth: string } = $props();
</script>

<p
  class={[
    'inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-medium',
    verdict === 'correct' ? 'bg-success-soft text-success' : 'bg-danger-soft text-danger',
  ]}
>
  <svg viewBox="0 0 16 16" class="size-4" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    {#if verdict === 'correct'}
      <path d="M3.5 8.5l3 3 6-7" />
    {:else}
      <path d="M4 4l8 8M12 4l-8 8" />
    {/if}
  </svg>
  {verdict === 'correct' ? 'Correct' : 'Wrong'}: ground truth is {truth}
</p>
