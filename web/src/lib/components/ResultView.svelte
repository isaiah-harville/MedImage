<script lang="ts">
import type { Prediction } from '#lib/contract.ts';
import { formatLabel, formatLatency, verdict } from '#lib/format.ts';
import ImagePanel from './ImagePanel.svelte';
import ProbabilityBars from './ProbabilityBars.svelte';
import SliceViewer from './SliceViewer.svelte';
import VerdictBadge from './VerdictBadge.svelte';

let { id, prediction, sourceLabel }: { id: string; prediction: Prediction; sourceLabel: string } =
  $props();

const outcome = $derived(verdict(prediction));
const predictedText = $derived(
  prediction.predicted.length === 0
    ? 'No finding above the threshold'
    : prediction.predicted.map(formatLabel).join(', '),
);
</script>

<section aria-labelledby={id} >
  <div class="rounded-[var(--radius-xl)] bg-[var(--color-viewer)] p-4 sm:p-5">
    <h2 {id} tabindex="-1" class="sr-only">Result for {sourceLabel}</h2>
    <p class="font-mono text-[11px] text-[var(--color-viewer-foreground)]">
      {sourceLabel} · {formatLatency(prediction.latencyMs)}
    </p>
    {#if prediction.slices}
      <div class="mt-3">
        <SliceViewer
          input={prediction.slices.input}
          attribution={prediction.slices.attribution}
          label={sourceLabel}
        />
      </div>
    {:else}
      <div class="mt-3 grid grid-cols-2 gap-3">
        <ImagePanel
          src={prediction.inputPng}
          alt="The scan as the model saw it, resized to the model's input size."
          caption="Input"
        />
        <ImagePanel
          src={prediction.attributionPng}
          alt="The same scan with an attribution map overlaid. Brighter regions influenced the top prediction more."
          caption="Attribution"
        />
      </div>
    {/if}
  </div>

  <div class="mt-4 rounded-[var(--radius-xl)] bg-card p-5 shadow-[var(--elevation-control)]">
    <div class="flex flex-wrap items-start justify-between gap-3">
      <div>
        <p class="text-[12px] text-foreground-muted">Model output</p>
        <p class="mt-0.5 text-[20px] leading-tight font-semibold tracking-[-0.015em]">{predictedText}</p>
      </div>
      {#if outcome !== 'unknown' && prediction.truth !== null}
        <VerdictBadge verdict={outcome} truth={formatLabel(prediction.truth)} />
      {/if}
    </div>
    <div class="mt-5">
      <ProbabilityBars {prediction} />
    </div>
    {#if prediction.threshold !== null}
      <p class="mt-3 text-[12px] text-foreground-muted">
        Decision threshold <span class="font-mono tabular-nums">{prediction.threshold.toFixed(2)}</span>,
        chosen on validation data.
      </p>
    {/if}
    <p class="mt-3 border-t border-border pt-3 text-[12px] text-foreground-muted">
      The attribution map shows the pixels or voxels that most influenced the top prediction, by Integrated
      Gradients. It is not evidence that the model uses clinically relevant features.
      {prediction.disclaimer}
    </p>
  </div>
</section>
