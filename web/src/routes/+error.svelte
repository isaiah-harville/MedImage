<script lang="ts">
import { page } from '$app/state';
</script>

<svelte:head>
  <title>{page.status} | OpenDiagnostics demo</title>
</svelte:head>

<div class="max-w-2xl">
  <p class="font-mono text-sm text-fg-secondary">{page.status}</p>
  <h1 class="mt-1 text-3xl font-semibold tracking-tight">
    {page.status === 404 ? 'Page not found' : 'Something went wrong'}
  </h1>
  <p class="mt-3 text-fg-secondary">{page.error?.message}</p>
  <p class="mt-6">
    <a href="/" class="font-medium text-accent underline underline-offset-2 hover:text-accent-hover"
      >Back to all scan types</a
    >
  </p>
</div>
