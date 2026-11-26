<script lang="ts">
let {
  allowUploads = false,
  detail,
  compact = false,
}: { allowUploads?: boolean; detail?: string; compact?: boolean } = $props();
</script>

<aside
  aria-label="Disclaimer"
  class={[
    'rounded-xl border border-warning-line bg-warning-soft text-warning',
    compact ? 'px-4 py-3 text-sm' : 'px-5 py-4',
  ]}
>
  <p class="font-semibold">Research models, not a medical device.</p>
  <p class={compact ? 'mt-1' : 'mt-1.5'}>
    Do not use these results for diagnosis or treatment. The models can be wrong, and nobody has
    validated them for clinical use.
    {#if allowUploads}
      Do not upload images that show names, dates of birth, record numbers, or any other patient
      identifier.
    {/if}
  </p>
  {#if detail}
    <p class="mt-1">Model service: {detail}</p>
  {/if}
</aside>
