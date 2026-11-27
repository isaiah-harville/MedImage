<script lang="ts">
import Disclaimer from '#lib/components/Disclaimer.svelte';
import ModelCard from '#lib/components/ModelCard.svelte';
import type { PageData } from './$types';

let { data }: { data: PageData } = $props();

const count = $derived(data.catalog.kind === 'ready' ? data.catalog.models.length : 0);
</script>

<svelte:head>
  <title>OpenDiagnostics demo</title>
  <meta
    name="description"
    content="Try open research models for medical image classification on public demo images."
  />
</svelte:head>

<section class="max-w-3xl">
  <p class="text-[13px] font-medium text-primary">Open models, public data, honest intervals</p>
  <h1 class="mt-3 text-[34px] leading-[1.1] font-semibold tracking-[-0.025em] text-balance sm:text-[44px]">
    See what a medical imaging model sees.
  </h1>
  <p class="mt-4 max-w-2xl text-[16px] leading-relaxed text-foreground-muted text-pretty">
    OpenDiagnostics trains image classifiers on public medical imaging benchmarks and publishes
    their weights, test scores, and 95% confidence intervals. Pick a scan type, run a model on an
    image from its test set or on your own, and see which pixels drove the prediction.
  </p>
  {#if count > 0}
    <p class="mt-5 text-[13px] text-foreground-muted">
      <span class="font-mono font-medium text-foreground tabular-nums">{count}</span> models,
      each trained on a public MedMNIST benchmark at 224 pixels.
    </p>
  {/if}
</section>

<div class="mt-8 max-w-3xl">
  <Disclaimer allowUploads={data.allowUploads} />
</div>

<section id="models" aria-labelledby="models-heading" class="mt-16 scroll-mt-24">
  <div class="flex flex-wrap items-end justify-between gap-3">
    <h2 id="models-heading" class="text-[20px] font-semibold tracking-[-0.015em]">Scan types</h2>
    <p class="text-[13px] text-foreground-muted">
      AUROC on each benchmark's held-out test split. Intervals are 95% bootstrap.
    </p>
  </div>
  {#if data.catalog.kind === 'unavailable'}
    <p role="alert" class="mt-6 rounded-[var(--radius-lg)] bg-error-soft px-4 py-4 text-error">
      The model service is not responding, so the list of models could not be loaded. Reload the page
      in a minute.
    </p>
  {:else if data.catalog.models.length === 0}
    <p class="mt-6 rounded-[var(--radius-lg)] bg-secondary px-4 py-6 text-foreground-muted">
      No models are being served yet. Models appear here once a run is published.
    </p>
  {:else}
    <ul class="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {#each data.catalog.models as model, index (model.id)}
        <li class="reveal flex" style:--delay="{Math.min(index, 7) * 40}ms"><ModelCard {model} /></li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  @keyframes rise {
    from {
      opacity: 0;
      transform: translateY(6px);
    }
  }

  .reveal {
    animation: rise 320ms var(--ease-out) both;
    animation-delay: var(--delay);
  }
</style>
