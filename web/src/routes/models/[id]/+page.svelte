<script lang="ts">
import { tick } from 'svelte';
import DemoGallery from '#lib/components/DemoGallery.svelte';
import ModelFacts from '#lib/components/ModelFacts.svelte';
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
  if (result.sampleId === null) return 'Your image';
  const index = data.demos.findIndex((d) => d.sampleId === result.sampleId);
  return index === -1 ? 'Test case' : `Case ${String(index + 1).padStart(2, '0')}`;
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
  <title>{data.model.scanType} | OpenDiagnostics</title>
</svelte:head>

<nav aria-label="Breadcrumb" class="text-[12px] text-foreground-muted">
  <a href="/" class="hover:text-foreground hover:underline underline-offset-4">Model catalog</a>
  <span aria-hidden="true" class="mx-1.5">/</span>
  <span aria-current="page">{data.model.scanType}</span>
</nav>

<header class="mt-3">
  <p class="text-[12px] text-foreground-muted">{data.model.modality}</p>
  <h1 class="text-[28px] leading-tight font-semibold tracking-[-0.02em]">{data.model.scanType}</h1>
  <p class="mt-1 text-[14px] text-foreground-muted">{data.model.question}</p>
</header>

<div class="mt-5">
  <ModelFacts model={data.model} />
</div>

<p class="sr-only" aria-live="polite">{status}</p>

<div class="mt-8 grid items-start gap-8 lg:grid-cols-[minmax(0,8fr)_minmax(0,4fr)]">
  <div class="min-w-0">
    {#if result}
      <ResultView prediction={result.prediction} {sourceLabel} />
    {:else}
      <div
        class="hidden aspect-[2/1] place-items-center rounded-[var(--radius-xl)] bg-[var(--color-viewer)] px-6 text-center lg:grid"
      >
        <div class="text-[var(--color-viewer-foreground)]">
          {#if pending}
            <p class="text-[14px] font-medium">Running the model</p>
            <p class="mt-1 text-[12px] opacity-75">
              Computing the prediction and its attribution map. This takes up to a minute.
            </p>
          {:else}
            <p class="text-[14px] font-medium">No case selected</p>
            <p class="mt-1 text-[12px] opacity-75">
              Choose a test case, or upload an image, to see the model's output here.
            </p>
          {/if}
        </div>
      </div>
    {/if}
  </div>

  <aside class="space-y-6" aria-label="Cases and model files">
    <section aria-labelledby="gallery-heading" class="rounded-[var(--radius-xl)] bg-card p-4 shadow-[var(--elevation-control)]">
      <h2 id="gallery-heading" class="text-[14px] font-semibold">Test cases</h2>
      <p class="mt-0.5 mb-3 text-[12px] text-foreground-muted">
        From the {data.model.dataset} test split, with each image's reference label.
      </p>
      {#if failure?.source === 'demo'}
        <p class="mb-3 rounded-[var(--radius-md)] bg-error-soft px-3 py-2 text-[12px] text-error">
          {failure.message}
        </p>
      {/if}
      <DemoGallery
        modelId={data.model.id}
        demos={data.demos}
        selected={result?.sampleId ?? null}
        pending={pending?.source === 'demo' ? pending.sampleId : null}
        onsubmit={submit}
      />
    </section>

    {#if data.allowUploads}
      <section aria-labelledby="upload-heading" class="rounded-[var(--radius-xl)] bg-card p-4 shadow-[var(--elevation-control)]">
        <h2 id="upload-heading" class="text-[14px] font-semibold">Your own image</h2>
        <p class="mt-0.5 mb-3 text-[12px] text-foreground-muted">
          A {data.model.scanType.toLowerCase()} with no names, dates, or record numbers in the
          pixels.
        </p>
        <UploadForm
          pending={pending?.source === 'upload'}
          serverError={failure?.source === 'upload' ? failure.message : null}
          onsubmit={submit}
        />
      </section>
    {/if}

    {#if data.model.weights}
      <WeightsDownload weights={data.model.weights} license={data.model.license} />
    {/if}
  </aside>
</div>
