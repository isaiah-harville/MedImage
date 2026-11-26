<script lang="ts">
import type { Prediction } from '#lib/contract.ts';
import { formatLabel, formatLatency, verdict } from '#lib/format.ts';
import Disclaimer from './Disclaimer.svelte';
import ImagePanel from './ImagePanel.svelte';
import ProbabilityBars from './ProbabilityBars.svelte';
import VerdictBadge from './VerdictBadge.svelte';

let {
  prediction,
  sourceLabel,
  allowUploads,
}: { prediction: Prediction; sourceLabel: string; allowUploads: boolean } = $props();

const outcome = $derived(verdict(prediction));
const predictedText = $derived(
  prediction.predicted.length === 0
    ? 'No finding above the threshold'
    : prediction.predicted.map(formatLabel).join(', '),
);
</script>

<section aria-labelledby="result-heading" class="rounded-2xl bg-surface p-5 shadow-card sm:p-6">
  <h2 id="result-heading" tabindex="-1" class="text-xl font-semibold tracking-tight">
    Result for {sourceLabel}
  </h2>

  <div class="mt-5 grid gap-8 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
    <div>
      <div class="grid grid-cols-2 gap-3 sm:gap-4">
        <ImagePanel
          src={prediction.inputPng}
          alt="The scan as the model saw it, resized to 224 by 224 pixels."
          caption="Input, as the model saw it"
        />
        <ImagePanel
          src={prediction.attributionPng}
          alt="The same scan with an attribution map overlaid. Brighter regions influenced the top prediction more."
          caption="Attribution map"
        />
      </div>
      <p class="mt-3 text-sm text-fg-secondary">
        The attribution map highlights the pixels that most influenced the top prediction. It is
        not proof that the model uses clinically relevant features.
      </p>
    </div>

    <div class="space-y-5">
      <div>
        <p class="text-sm text-fg-secondary">Predicted</p>
        <p class="text-2xl font-semibold tracking-tight">{predictedText}</p>
      </div>
      {#if outcome !== 'unknown' && prediction.truth !== null}
        <VerdictBadge verdict={outcome} truth={formatLabel(prediction.truth)} />
      {/if}
      <ProbabilityBars {prediction} />
      <dl class="flex flex-wrap gap-x-6 gap-y-1 text-sm text-fg-secondary">
        {#if prediction.threshold !== null}
          <div class="flex gap-1.5">
            <dt>Decision threshold</dt>
            <dd class="font-mono tabular-nums text-fg">{prediction.threshold.toFixed(2)}</dd>
          </div>
        {/if}
        <div class="flex gap-1.5">
          <dt>Latency</dt>
          <dd class="font-mono tabular-nums text-fg">{formatLatency(prediction.latencyMs)}</dd>
        </div>
      </dl>
    </div>
  </div>

  <div class="mt-6">
    <Disclaimer compact {allowUploads} detail={prediction.disclaimer} />
  </div>
</section>
