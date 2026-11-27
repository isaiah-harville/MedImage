<script lang="ts">
import { page } from '$app/state';
</script>

<svelte:head>
  <title>{page.status} | OpenDiagnostics demo</title>
</svelte:head>

<div class="max-w-2xl">
  <p class="font-mono text-sm text-foreground-muted">{page.status}</p>
  <h1 class="mt-1 text-3xl font-semibold tracking-tight">
    {page.status === 404 ? 'Page not found' : 'Something went wrong'}
  </h1>
  <p class="mt-3 text-foreground-muted">{page.error?.message}</p>
  <p class="mt-6">
    <a href="/" class="font-medium text-primary underline underline-offset-2 hover:opacity-80"
      >Back to all scan types</a
    >
  </p>
</div>
