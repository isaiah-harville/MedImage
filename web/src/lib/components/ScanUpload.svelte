<script lang="ts">
import { Button } from '@sivir-ui/svelte/components/button';
import { ACCEPTED_UPLOAD_TYPES, checkUpload } from '#lib/upload.ts';

export type QueuedScan = {
  id: number;
  name: string;
  status: 'waiting' | 'running' | 'done' | 'failed';
  message?: string;
};

let {
  scanType,
  busy,
  queue,
  onrun,
}: {
  scanType: string;
  busy: boolean;
  queue: QueuedScan[];
  onrun: (files: File[]) => void;
} = $props();

const MAX_FILES = 20;
let files = $state<File[]>([]);
let consented = $state(false);
let dragging = $state(false);
let input = $state<HTMLInputElement>();

const problems = $derived(
  files.flatMap((file) => {
    const check = checkUpload(file);
    return check.ok ? [] : [`${file.name}: ${check.message}`];
  }),
);
const tooMany = $derived(files.length > MAX_FILES);

function choose(list: FileList | null | undefined) {
  files = list ? [...list] : [];
}

function submit(event: SubmitEvent) {
  event.preventDefault();
  if (files.length === 0 || problems.length > 0 || tooMany || !consented) return;
  onrun(files);
  files = [];
  if (input) input.value = '';
}

const STATUS: Record<QueuedScan['status'], string> = {
  waiting: 'Waiting',
  running: 'Running',
  done: 'Done',
  failed: 'Failed',
};
</script>

<form
  method="POST"
  action="?/upload"
  enctype="multipart/form-data"
  onsubmit={submit}
  class="rounded-[var(--radius-xl)] bg-card p-5 shadow-[var(--elevation-control)]"
>
  <h2 id="upload-heading" class="text-[14px] font-semibold">Run your scans</h2>
  <p class="mt-0.5 text-[12px] text-foreground-muted">
    {scanType} images as PNG or JPEG, up to {MAX_FILES} at a time and 10 MB each. They run one
    after another, in server memory only, and are never stored.
    <a href="/privacy" class="text-primary underline-offset-4 hover:underline">How we handle images</a>
  </p>

  <label
    for="scan-files"
    ondragover={(event) => {
      event.preventDefault();
      dragging = true;
    }}
    ondragleave={() => (dragging = false)}
    ondrop={(event) => {
      event.preventDefault();
      dragging = false;
      choose(event.dataTransfer?.files);
    }}
    class={[
      'mt-4 flex cursor-pointer flex-col items-center justify-center rounded-[var(--radius-lg)] border border-dashed px-6 py-8 text-center transition-colors',
      dragging ? 'border-primary bg-accent-tint' : 'border-border-strong hover:bg-secondary/60',
    ]}
  >
    <span class="text-[13px] font-medium">
      {files.length === 0
        ? 'Drop scans here or choose files'
        : `${files.length} ${files.length === 1 ? 'scan' : 'scans'} selected`}
    </span>
    <span class="mt-1 text-[12px] text-foreground-muted">
      {files.length === 0 ? 'No names, dates, or record numbers in the pixels.' : files.map((f) => f.name).join(', ')}
    </span>
    <input
      bind:this={input}
      id="scan-files"
      name="file"
      type="file"
      multiple
      required
      accept={ACCEPTED_UPLOAD_TYPES.join(',')}
      aria-describedby="scan-errors"
      onchange={(event) => choose(event.currentTarget.files)}
      class="sr-only"
    />
  </label>

  <div id="scan-errors" aria-live="polite" class="mt-2 space-y-0.5 text-[12px] text-error">
    {#if tooMany}<p>Choose at most {MAX_FILES} scans at a time.</p>{/if}
    {#each problems as problem (problem)}<p>{problem}</p>{/each}
  </div>

  <div class="mt-3 flex flex-wrap items-end justify-between gap-4">
    <label class="flex max-w-md items-start gap-3 text-[12px]">
      <input
        type="checkbox"
        name="consent"
        value="yes"
        required
        bind:checked={consented}
        class="mt-0.5 size-4 shrink-0 accent-[var(--color-primary)]"
      />
      <span>
        I have removed names and other identifiers from these images, I have the right to share
        them, and I understand the results are not medical advice.
      </span>
    </label>
    <Button
      type="submit"
      variant="primary"
      disabled={busy || files.length === 0 || problems.length > 0 || tooMany || !consented}
    >
      {files.length > 1 ? `Run ${files.length} scans` : 'Run the model'}
    </Button>
  </div>

  {#if queue.length > 0}
    <ol class="mt-4 divide-y divide-border border-t border-border text-[12px]" aria-label="Scan queue">
      {#each queue as item (item.id)}
        <li class="flex items-center justify-between gap-4 py-2">
          <span class="min-w-0 truncate">{item.name}</span>
          <span
            class={[
              'shrink-0 font-medium',
              item.status === 'failed' && 'text-error',
              item.status === 'running' && 'text-primary',
              item.status === 'done' && 'text-foreground-muted',
              item.status === 'waiting' && 'text-foreground-muted',
            ]}
          >
            {item.status === 'failed' && item.message ? item.message : STATUS[item.status]}
          </span>
        </li>
      {/each}
    </ol>
  {/if}
</form>
