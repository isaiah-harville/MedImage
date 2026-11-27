<script lang="ts">
import ModelCatalog from '#lib/components/ModelCatalog.svelte';
import type { PageData } from './$types';

let { data }: { data: PageData } = $props();
</script>

<svelte:head>
  <title>Model catalog | OpenDiagnostics</title>
  <meta
    name="description"
    content="Open research models for medical image classification, with test AUROC and 95% confidence intervals."
  />
</svelte:head>

<header class="grid gap-6 border-b border-border pb-8 md:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] md:items-end">
  <div>
    <h1 class="text-[28px] leading-tight font-semibold tracking-[-0.02em]">Model catalog</h1>
    <p class="mt-2 max-w-2xl text-[14px] leading-relaxed text-foreground-muted">
      Image classifiers trained on public medical imaging benchmarks. Each model reports its test
      AUROC with a 95% bootstrap interval, and its weights, predictions, and training run are
      public. Open a model to run it on a test image or your own, with an attribution map of the
      pixels that drove its prediction.
    </p>
  </div>
  <dl class="grid grid-cols-3 gap-4 text-[12px] md:justify-self-end">
    <div>
      <dt class="text-foreground-muted">Models</dt>
      <dd class="mt-0.5 font-mono text-[18px] font-medium tabular-nums">
        {data.catalog.kind === 'ready' ? data.catalog.models.length : 0}
      </dd>
    </div>
    <div>
      <dt class="text-foreground-muted">Benchmark</dt>
      <dd class="mt-0.5 text-[14px] font-medium">MedMNIST v2</dd>
    </div>
    <div>
      <dt class="text-foreground-muted">Input</dt>
      <dd class="mt-0.5 text-[14px] font-medium">224 x 224 px</dd>
    </div>
  </dl>
</header>

<section aria-labelledby="models-heading" class="mt-8">
  <h2 id="models-heading" class="sr-only">Models</h2>
  {#if data.catalog.kind === 'unavailable'}
    <p role="alert" class="rounded-[var(--radius-lg)] bg-error-soft px-4 py-4 text-error">
      The model service is not responding, so the catalog could not be loaded. Reload the page in a
      minute.
    </p>
  {:else if data.catalog.models.length === 0}
    <p class="rounded-[var(--radius-lg)] bg-secondary px-4 py-6 text-foreground-muted">
      No models are being served yet. A model appears here once its training run is published.
    </p>
  {:else}
    <ModelCatalog models={data.catalog.models} />
    <p class="mt-3 text-[12px] text-foreground-muted">
      AUROC is measured on each benchmark's held-out test split. For multiclass and multilabel
      tasks it is the mean over classes.
    </p>
  {/if}
</section>
