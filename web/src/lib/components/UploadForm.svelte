<script lang="ts">
import { Button } from '@sivir-ui/svelte/components/button';
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

<form method="POST" action="?/upload" enctype="multipart/form-data" use:enhance={onsubmit}>
  <label for="upload-file" class="text-[13px] font-medium">Image file</label>
  <p id="upload-hint" class="mt-1 text-[12px] text-foreground-muted">
    PNG or JPEG, up to 10 MB and 4096 x 4096 pixels. It is processed in server memory for this
    request only and never stored.
    <a href="/privacy" class="text-primary underline-offset-4 hover:underline">How we handle your image</a>
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
    class="mt-3 block w-full text-[13px] text-foreground-muted file:mr-3 file:rounded-[var(--radius-md)] file:border-0 file:bg-secondary file:px-3 file:py-2 file:text-[13px] file:font-medium file:text-foreground hover:file:bg-border"
  />
  <p id="upload-error" aria-live="polite" class="mt-2 min-h-5 text-[12px] text-error">
    {message ?? ''}
  </p>
  <label class="mb-4 flex items-start gap-3 text-[13px]">
    <input
      type="checkbox"
      name="consent"
      value="yes"
      required
      bind:checked={consented}
      class="mt-0.5 size-4 shrink-0 accent-[var(--color-primary)]"
    />
    <span>
      I have removed names and other identifiers from this image, I have the right to share it, and
      I understand the result is not medical advice.
    </span>
  </label>
  <Button
    type="submit"
    variant="primary"
    loading={pending}
    loadingLabel="Running the model"
    disabled={clientError !== null || (hydrated && !consented)}
  >
    Run the model
  </Button>
</form>
