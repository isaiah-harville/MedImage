<script lang="ts">
import { tick, untrack } from 'svelte';
import DemoGallery from '#lib/components/DemoGallery.svelte';
import ModelFacts from '#lib/components/ModelFacts.svelte';
import ResultView from '#lib/components/ResultView.svelte';
import ScanUpload, { type QueuedScan } from '#lib/components/ScanUpload.svelte';
import WeightsDownload from '#lib/components/WeightsDownload.svelte';
import type { Prediction } from '#lib/contract.ts';
import type { SubmitFunction } from '$app/forms';
import { deserialize } from '$app/forms';
import type { ActionData, PageData } from './$types';

let { data, form }: { data: PageData; form: ActionData } = $props();

type Entry = { key: number; label: string; sampleId: string | null; prediction: Prediction };

let nextKey = 0;
const caseLabel = (sampleId: string) => {
  const index = data.demos.findIndex((d) => d.sampleId === sampleId);
  return index === -1 ? 'Test case' : `Case ${String(index + 1).padStart(2, '0')}`;
};

// Without JavaScript the form posts normally and the page renders the one result; with it,
// every run goes through call() below. Either way the list starts from that first result.
const initial = untrack(() => form);
let results = $state<Entry[]>(
  initial?.prediction
    ? [
        {
          key: nextKey++,
          label: initial.sampleId ? caseLabel(initial.sampleId) : 'Your scan',
          sampleId: initial.sampleId ?? null,
          prediction: initial.prediction,
        },
      ]
    : [],
);
let queue = $state<QueuedScan[]>([]);
let busy = $state(false);
let runningCase = $state<string | null>(null);
let demoError = $state<string | null>(
  initial?.source === 'demo' ? (initial.message ?? null) : null,
);
let status = $state('');

type Outcome = { prediction: Prediction } | { message: string };

async function call(action: 'demo' | 'upload', body: FormData): Promise<Outcome> {
  try {
    const response = await fetch(`?/${action}`, {
      method: 'POST',
      body,
      headers: { 'x-sveltekit-action': 'true', accept: 'application/json' },
    });
    const result = deserialize(await response.text());
    if (result.type === 'success' && result.data?.prediction) {
      return { prediction: result.data.prediction as Prediction };
    }
    if (result.type === 'failure' && typeof result.data?.message === 'string') {
      return { message: result.data.message };
    }
  } catch {
    // Network failure: reported below like any other unanswered request.
  }
  return { message: 'The model service did not answer. Try again in a minute.' };
}

async function show(entry: Omit<Entry, 'key'>) {
  results = [{ key: nextKey++, ...entry }, ...results];
  await tick();
  document.getElementById(`result-${results[0]?.key}`)?.focus();
}

async function runScans(files: File[]) {
  busy = true;
  const items = files.map((file) => ({
    id: nextKey++,
    name: file.name,
    status: 'waiting' as const,
  }));
  queue = [...items, ...queue];
  for (const [index, file] of files.entries()) {
    const item = items[index];
    if (!item) continue;
    const set = (patch: Partial<QueuedScan>) => {
      queue = queue.map((q) => (q.id === item.id ? { ...q, ...patch } : q));
    };
    set({ status: 'running' });
    status = `Running scan ${index + 1} of ${files.length}.`;
    const body = new FormData();
    body.set('file', file);
    body.set('consent', 'yes');
    const outcome = await call('upload', body);
    if ('prediction' in outcome) {
      set({ status: 'done' });
      await show({ label: file.name, sampleId: null, prediction: outcome.prediction });
    } else {
      set({ status: 'failed', message: outcome.message });
    }
  }
  status = `Finished ${files.length} ${files.length === 1 ? 'scan' : 'scans'}.`;
  busy = false;
}

const runCase: SubmitFunction = ({ formData, cancel }) => {
  cancel();
  const sampleId = formData.get('sample_id');
  if (typeof sampleId !== 'string' || busy) return;
  busy = true;
  runningCase = sampleId;
  demoError = null;
  status = `Running ${caseLabel(sampleId)}.`;
  call('demo', formData).then(async (outcome) => {
    if ('prediction' in outcome) {
      await show({ label: caseLabel(sampleId), sampleId, prediction: outcome.prediction });
      status = `${caseLabel(sampleId)} finished.`;
    } else {
      demoError = outcome.message;
      status = outcome.message;
    }
    runningCase = null;
    busy = false;
  });
};

const selected = $derived(results.find((r) => r.sampleId !== null)?.sampleId ?? null);
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
  <div class="min-w-0 space-y-6">
    {#if data.allowUploads}
      <section aria-labelledby="upload-heading">
        <ScanUpload scanType={data.model.scanType} {busy} {queue} onrun={runScans} />
      </section>
    {/if}

    {#if results.length === 0}
      <div
        class="hidden aspect-[2/1] place-items-center rounded-[var(--radius-xl)] bg-[var(--color-viewer)] px-6 text-center lg:grid"
      >
        <div class="text-[var(--color-viewer-foreground)]">
          <p class="text-[14px] font-medium">{busy ? 'Running the model' : 'No results yet'}</p>
          <p class="mt-1 text-[12px] opacity-75">
            {busy
              ? 'Computing the prediction and its attribution map. Each scan takes up to a minute.'
              : 'Results appear here, newest first, as each scan or test case finishes.'}
          </p>
        </div>
      </div>
    {:else}
      <div class="space-y-8" aria-label="Results, newest first">
        {#each results as entry (entry.key)}
          <ResultView id="result-{entry.key}" prediction={entry.prediction} sourceLabel={entry.label} />
        {/each}
      </div>
    {/if}
  </div>

  <aside class="space-y-6" aria-label="Test cases and model files">
    <section aria-labelledby="gallery-heading" class="rounded-[var(--radius-xl)] bg-card p-4 shadow-[var(--elevation-control)]">
      <h2 id="gallery-heading" class="text-[14px] font-semibold">Test cases</h2>
      <p class="mt-0.5 mb-3 text-[12px] text-foreground-muted">
        From the {data.model.dataset} test split, with each image's reference label.
      </p>
      {#if demoError}
        <p class="mb-3 rounded-[var(--radius-md)] bg-error-soft px-3 py-2 text-[12px] text-error">
          {demoError}
        </p>
      {/if}
      <DemoGallery
        modelId={data.model.id}
        demos={data.demos}
        {selected}
        pending={runningCase}
        disabled={busy}
        onsubmit={runCase}
      />
    </section>

    {#if data.model.weights}
      <WeightsDownload weights={data.model.weights} license={data.model.license} />
    {/if}
  </aside>
</div>
