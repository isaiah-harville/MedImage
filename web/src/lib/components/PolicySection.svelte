<script lang="ts">
import type { Snippet } from 'svelte';

let { title, children }: { title: string; children: Snippet } = $props();
</script>

<section class="mt-10">
  <h2 class="text-xl font-semibold tracking-tight">{title}</h2>
  <div class="mt-3 space-y-3 text-foreground-muted [&_strong]:font-medium [&_strong]:text-foreground">
    {@render children()}
  </div>
</section>
