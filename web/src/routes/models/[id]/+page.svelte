<script lang="ts">
import { Badge } from '@sivir-ui/svelte/components/badge';
import { tick } from 'svelte';
import AurocStat from '#lib/components/AurocStat.svelte';
import DemoGallery from '#lib/components/DemoGallery.svelte';
import Disclaimer from '#lib/components/Disclaimer.svelte';
import ResultView from '#lib/components/ResultView.svelte';
import UploadForm from '#lib/components/UploadForm.svelte';
import WeightsDownload from '#lib/components/WeightsDownload.svelte';
import { formatLabel } from '#lib/format.ts';
import type { SubmitFunction } from '$app/forms';
import type { ActionData, PageData } from './$types';

let { data, form }: { data: PageData; form: ActionData } = $props();

type Pending = { source: 'demo'; sampleId: string } | { source: 'upload' } | null;
let pending = $state<Pending>(null);

const result = $derived(
  form?.prediction ? { prediction: form.prediction, sampleId: form.sampleId ?? null } : null,
);
const failure = $derived(
  form?.message && form.source ? { source: form.source, message: form.message } : null,
);
const sourceLabel = $derived.by(() => {
  if (!result) return '';
  if (result.sampleId === null) return 'your image';
  const index = data.demos.findIndex((d) => d.sampleId === result.sampleId);
  return index === -1 ? 'the demo image' : `demo image ${index + 1}`;
});
const status = $derived.by(() => {
  if (pending?.source === 'demo') return 'Running the model on the demo image.';
  if (pending?.source === 'upload') return 'Running the model on your image.';
  if (failure) return failure.message;
  return '';
});

const submit: SubmitFunction = ({ formData, action }) => {
  const sampleId = formData.get('sample_id');
  pending =
    action.search === '?/demo' && typeof sampleId === 'string'
      ? { source: 'demo', sampleId }
      : { source: 'upload' };
  return async ({ update, result: actionResult }) => {
    await update({ reset: false });
    pending = null;
    if (actionResult.type === 'success') {
      await tick();
      document.getElementById('result-heading')?.focus();
    }
  };
};
</script>

<svelte:head>
  <title>{data.model.scanType} | OpenDiagnostics demo</title>
</svelte:head>

<nav aria-label="Breadcrumb" class="text-[13px] text-foreground-muted">
  <a href="/#models" class="underline-offset-4 hover:text-foreground hover:underline">Scan types</a>
  <span aria-hidden="true" class="mx-1.5">/</span>
  <span aria-current="page" class="text-foreground">{data.model.scanType}</span>
</nav>

<header class="mt-4 flex flex-wrap items-end justify-between gap-6">
  <div class="max-w-2xl">
    <div class="flex flex-wrap gap-2">
      <Badge variant="secondary">{data.model.modality}</Badge>
      <Badge variant="outline">{data.model.license}</Badge>
    </div>
    <h1 class="mt-3 text-[32px] leading-[1.1] font-semibold tracking-[-0.025em] text-balance">
      {data.model.scanType}
    </h1>
    <p class="mt-2 text-[16px] text-foreground-muted">{data.model.question}</p>
    <p class="mt-1 text-[13px] text-foreground-muted">
      Classes: {data.model.classes.map(formatLabel).join(', ')}. Trained on {data.model.dataset}.
    </p>
  </div>
  <dl><AurocStat auroc={data.model.testAuroc} large /></dl>
</header>

<div class="mt-6">
  <Disclaimer allowUploads={data.allowUploads} />
</div>

<p class="sr-only" aria-live="polite">{status}</p>

<div class="mt-10 grid items-start gap-8 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
  <div class="order-2 space-y-10 lg:order-1">
    <section aria-labelledby="gallery-heading">
      <h2 id="gallery-heading" class="text-[20px] font-semibold tracking-[-0.015em]">Demo images</h2>
      <p class="mt-1 text-[13px] text-foreground-muted">
        From the {data.model.dataset} test split, with the dataset's label. Choose one to run the
        model.
      </p>
      {#if failure?.source === 'demo'}
        <p class="mt-4 rounded-[var(--radius-lg)] bg-error-soft px-4 py-3 text-[13px] text-error">
          {failure.message}
        </p>
      {/if}
      <div class="mt-4">
        <DemoGallery
          modelId={data.model.id}
          demos={data.demos}
          selected={result?.sampleId ?? null}
          pending={pending?.source === 'demo' ? pending.sampleId : null}
          onsubmit={submit}
        />
      </div>
    </section>

    {#if data.allowUploads}
      <section aria-labelledby="upload-heading">
        <h2 id="upload-heading" class="text-[20px] font-semibold tracking-[-0.015em]">Your own image</h2>
        <p class="mt-1 text-[13px] text-foreground-muted">
          Upload a {data.model.scanType.toLowerCase()} with no patient identifiers in the pixels.
        </p>
        <div class="mt-4 rounded-[var(--radius-xl)] bg-card p-5 shadow-[var(--elevation-control)]">
          <UploadForm
            pending={pending?.source === 'upload'}
            serverError={failure?.source === 'upload' ? failure.message : null}
            onsubmit={submit}
          />
        </div>
      </section>
    {/if}
  </div>

  <aside class="order-1 space-y-6 lg:sticky lg:top-24 lg:order-2" aria-label="Result and weights">
    {#if result}
      <ResultView prediction={result.prediction} {sourceLabel} allowUploads={data.allowUploads} />
    {:else}
      <div
        class="rounded-[var(--radius-xl)] border border-dashed border-border-strong px-6 py-10 text-center text-[13px] text-foreground-muted"
      >
        {#if pending}
          <p class="font-medium text-foreground">Running the model</p>
          <p class="mt-1">The prediction and its attribution map appear here in a second or two.</p>
        {:else}
          <p class="font-medium text-foreground">No result yet</p>
          <p class="mt-1">Run a demo image or your own, and the prediction appears here.</p>
        {/if}
      </div>
    {/if}
    {#if data.model.weights}
      <section aria-labelledby="weights-heading">
        <WeightsDownload weights={data.model.weights} license={data.model.license} />
      </section>
    {/if}
  </aside>
</div>
