<script lang="ts">
import { ACCEPTED_UPLOAD_TYPES, checkUpload } from '#lib/upload.ts';
import type { SubmitFunction } from '$app/forms';
import { enhance } from '$app/forms';

let {
  pending,
  serverError,
  onsubmit,
}: { pending: boolean; serverError: string | null; onsubmit: SubmitFunction } = $props();

let clientError = $state<string | null>(null);
let consented = $state(false);
let hydrated = $state(false);
$effect(() => {
  hydrated = true;
});
const message = $derived(clientError ?? serverError);

function validate(event: Event & { currentTarget: HTMLInputElement }) {
  const file = event.currentTarget.files?.[0];
  const check = file ? checkUpload(file) : { ok: true as const };
  clientError = check.ok ? null : check.message;
  event.currentTarget.setCustomValidity(clientError ?? '');
}
</script>

<form
  method="POST"
  action="?/upload"
  enctype="multipart/form-data"
  use:enhance={onsubmit}
  class="rounded-2xl bg-surface p-5 shadow-card"
>
  <label for="upload-file" class="font-medium">Your image</label>
  <p id="upload-hint" class="mt-1 text-sm text-fg-secondary">
    PNG or JPEG, up to 10 MB and 4096 x 4096 pixels. The image is processed in server memory for
    this request only and is never stored.
    <a href="/privacy" class="text-accent underline underline-offset-2 hover:text-accent-hover"
      >How we handle your image</a
    >
  </p>
  <input
    id="upload-file"
    name="file"
    type="file"
    required
    accept={ACCEPTED_UPLOAD_TYPES.join(',')}
    aria-describedby="upload-hint upload-error"
    aria-invalid={message !== null}
    onchange={validate}
    class="mt-3 block w-full text-sm text-fg-secondary file:mr-3 file:rounded-lg file:border-0 file:bg-muted file:px-3 file:py-2 file:font-medium file:text-fg hover:file:bg-line"
  />
  <p id="upload-error" aria-live="polite" class="mt-2 min-h-5 text-sm text-danger">
    {message ?? ''}
  </p>
  <label class="mb-4 flex items-start gap-3 text-sm">
    <input
      type="checkbox"
      name="consent"
      value="yes"
      required
      bind:checked={consented}
      class="mt-0.5 size-4 shrink-0 accent-accent"
    />
    <span>
      I have removed names and other identifiers from this image, I have the right to share it, and
      I understand the result is not medical advice.
    </span>
  </label>
  <button
    type="submit"
    disabled={pending || clientError !== null || (hydrated && !consented)}
    class="mt-2 inline-flex h-10 items-center rounded-lg bg-accent px-4 text-sm font-medium text-accent-fg transition duration-150 ease-out hover:bg-accent-hover active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50"
  >
    {pending ? 'Running the model' : 'Run the model'}
  </button>
</form>
