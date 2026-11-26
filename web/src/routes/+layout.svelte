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
  class="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-10 focus:rounded-lg focus:bg-surface focus:px-3 focus:py-2"
  >Skip to content</a
>
<div class="flex min-h-dvh flex-col">
  <SiteHeader />
  <main id="main" class="mx-auto w-full max-w-6xl flex-1 px-4 pt-8 sm:px-6 sm:pt-12">
    {@render children()}
  </main>
  <SiteFooter {credits} />
</div>
