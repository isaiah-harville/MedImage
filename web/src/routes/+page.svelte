<script lang="ts">
import Disclaimer from '#lib/components/Disclaimer.svelte';
import ModelCard from '#lib/components/ModelCard.svelte';
import type { PageData } from './$types';

let { data }: { data: PageData } = $props();
</script>

<svelte:head>
  <title>OpenDiagnostics demo</title>
  <meta
    name="description"
    content="Try open research models for medical image classification on public demo images."
  />
</svelte:head>

<div class="max-w-3xl">
  <Disclaimer allowUploads={data.allowUploads} />
</div>

<div class="mt-8 max-w-3xl">
  <h1 class="text-3xl font-semibold tracking-tight sm:text-4xl">
    Try open research models on medical scans
  </h1>
  <p class="mt-3 text-lg text-fg-secondary">
    OpenDiagnostics trains image classifiers on public medical imaging benchmarks and reports their
    test scores with confidence intervals. Pick a scan type, run a model on an image from its test
    set, and see which pixels drove the prediction.
  </p>
</div>

<section aria-labelledby="models-heading" class="mt-12">
  <h2 id="models-heading" class="text-xl font-semibold tracking-tight">Scan types</h2>
  <p class="mt-1 text-sm text-fg-secondary">
    Test AUROC is measured on the benchmark's held-out test split. The interval is a 95% percentile
    bootstrap over patients.
  </p>
  {#if data.catalog.kind === 'unavailable'}
    <p role="alert" class="mt-6 rounded-xl bg-danger-soft px-4 py-4 text-danger">
      The model service is not responding, so the list of models could not be loaded. Reload the page
      in a minute.
    </p>
  {:else if data.catalog.models.length === 0}
    <p class="mt-6 rounded-xl bg-muted px-4 py-6 text-fg-secondary">
      No models are being served yet. Models appear here once the inference service loads a
      published run.
    </p>
  {:else}
    <ul class="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {#each data.catalog.models as model (model.id)}
        <li class="flex"><ModelCard {model} /></li>
      {/each}
    </ul>
  {/if}
</section>
