<script lang="ts">
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

<nav aria-label="Breadcrumb" class="text-sm text-fg-secondary">
  <a href="/" class="underline underline-offset-2 hover:text-fg">All scan types</a>
</nav>

<div class="mt-3 flex flex-wrap items-end justify-between gap-6">
  <div class="max-w-2xl">
    <h1 class="text-3xl font-semibold tracking-tight">{data.model.scanType}</h1>
    <p class="mt-2 text-lg">{data.model.question}</p>
    <p class="mt-1 text-sm text-fg-secondary">
      Classes: {data.model.classes.map(formatLabel).join(', ')}. Trained on {data.model.dataset} ({data
        .model.license}).
    </p>
  </div>
  <dl><AurocStat auroc={data.model.testAuroc} /></dl>
</div>

<div class="mt-6">
  <Disclaimer allowUploads={data.allowUploads} compact />
</div>

<p class="sr-only" aria-live="polite">{status}</p>

{#if result}
  <div class="mt-8">
    <ResultView prediction={result.prediction} {sourceLabel} allowUploads={data.allowUploads} />
  </div>
{/if}

<section aria-labelledby="gallery-heading" class="mt-10">
  <h2 id="gallery-heading" class="text-xl font-semibold tracking-tight">Demo images</h2>
  <p class="mt-1 text-sm text-fg-secondary">
    Images from the {data.model.dataset} test split, with the dataset's label. Choose one to run the
    model.
  </p>
  {#if failure?.source === 'demo'}
    <p class="mt-4 rounded-xl bg-danger-soft px-4 py-3 text-sm text-danger">{failure.message}</p>
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
  <section aria-labelledby="upload-heading" class="mt-10 max-w-xl">
    <h2 id="upload-heading" class="text-xl font-semibold tracking-tight">Your own image</h2>
    <p class="mt-1 text-sm text-fg-secondary">
      Upload a {data.model.scanType.toLowerCase()} with no patient identifiers in the pixels.
    </p>
    <div class="mt-4">
      <UploadForm
        pending={pending?.source === 'upload'}
        serverError={failure?.source === 'upload' ? failure.message : null}
        onsubmit={submit}
      />
    </div>
  </section>
{/if}

{#if data.model.weights}
  <WeightsDownload weights={data.model.weights} license={data.model.license} />
{/if}
