<script lang="ts">
let { allowUploads = false, detail }: { allowUploads?: boolean; detail?: string } = $props();
</script>

<aside
  aria-label="Disclaimer"
  class="flex gap-3 rounded-[var(--radius-lg)] bg-warning-soft px-4 py-3 text-[13px] text-foreground"
>
  <svg
    viewBox="0 0 24 24"
    class="mt-0.5 size-4 shrink-0 text-warning"
    aria-hidden="true"
    fill="none"
    stroke="currentColor"
    stroke-width="2.25"
    stroke-linecap="round"
    stroke-linejoin="round"
  >
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3" />
    <path d="M12 9v4" />
    <path d="M12 17h.01" />
  </svg>
  <div>
    <p class="font-semibold">Research models, not a medical device</p>
    <p class="mt-0.5 text-foreground-muted">
      Do not use these results for diagnosis or treatment. The models can be wrong, and nobody has
      validated them for clinical use.
      {#if allowUploads}
        Do not upload images that show names, dates of birth, record numbers, or any other patient
        identifier.
      {/if}
      {#if detail}
        <span class="mt-1 block">Model service: {detail}</span>
      {/if}
    </p>
  </div>
</aside>
