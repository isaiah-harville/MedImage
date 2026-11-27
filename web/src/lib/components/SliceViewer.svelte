<script lang="ts">
import { untrack } from 'svelte';
import ImagePanel from './ImagePanel.svelte';

let { input, attribution, label }: { input: string[]; attribution: string[]; label: string } =
  $props();

let depth = $state(untrack(() => Math.floor(input.length / 2)));
</script>

<div class="grid grid-cols-2 gap-3">
  <ImagePanel
    src={input[depth] ?? ''}
    alt="Slice {depth + 1} of {input.length} of the volume as the model saw it."
    caption="Input"
  />
  <ImagePanel
    src={attribution[depth] ?? ''}
    alt="Slice {depth + 1} of {input.length} with the attribution map overlaid. Brighter regions influenced the top prediction more."
    caption="Attribution"
  />
</div>
<label class="mt-3 flex items-center gap-3 font-mono text-[11px] text-[var(--color-viewer-foreground)]">
  <span class="shrink-0 uppercase">Slice</span>
  <input
    type="range"
    min="0"
    max={input.length - 1}
    bind:value={depth}
    aria-label="Slice of {label}"
    aria-valuetext="Slice {depth + 1} of {input.length}"
    class="h-1 w-full cursor-pointer accent-[var(--color-primary)]"
  />
  <span class="w-14 shrink-0 text-right tabular-nums">{depth + 1} / {input.length}</span>
</label>
