<script lang="ts">
import '../app.css';
import type { Snippet } from 'svelte';
import SiteFooter from '#lib/components/SiteFooter.svelte';
import SiteHeader from '#lib/components/SiteHeader.svelte';
import { datasetCredits } from '#lib/format.ts';
import type { LayoutData } from './$types';

let { data, children }: { data: LayoutData; children: Snippet } = $props();

const credits = $derived(data.catalog.kind === 'ready' ? datasetCredits(data.catalog.models) : []);
</script>

<a
  href="#main"
  class="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-10 focus:rounded-[var(--radius-md)] focus:bg-card focus:px-3 focus:py-2 focus:shadow-[var(--elevation-float)]"
  >Skip to content</a
>
<div class="flex min-h-dvh flex-col">
  <SiteHeader />
  <main id="main" class="mx-auto w-full max-w-6xl flex-1 px-5 pt-10 sm:pt-14">
    {@render children()}
  </main>
  <SiteFooter {credits} />
</div>
