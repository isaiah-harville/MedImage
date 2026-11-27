<script lang="ts">
import * as Card from '@sivir-ui/svelte/components/card';
import type { Prediction } from '#lib/contract.ts';
import { formatLabel, formatLatency, verdict } from '#lib/format.ts';
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

<Card.Root>
  <section aria-labelledby="result-heading" class="p-6">
    <h2 id="result-heading" tabindex="-1" class="text-[13px] font-medium text-foreground-muted focus:outline-none">
      Result for {sourceLabel}
    </h2>
    <p class="mt-1 text-[24px] leading-tight font-semibold tracking-[-0.02em]">{predictedText}</p>
    {#if outcome !== 'unknown' && prediction.truth !== null}
      <div class="mt-3">
        <VerdictBadge verdict={outcome} truth={formatLabel(prediction.truth)} />
      </div>
    {/if}

    <div class="mt-5 grid grid-cols-2 gap-3">
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
    <p class="mt-3 text-[12px] text-foreground-muted">
      The map highlights the pixels that most influenced the top prediction. It is not proof that
      the model uses clinically relevant features.
    </p>

    <div class="mt-6 border-t border-border pt-5">
      <ProbabilityBars {prediction} />
    </div>
    <dl class="mt-5 flex flex-wrap gap-x-6 gap-y-1 text-[12px] text-foreground-muted">
      {#if prediction.threshold !== null}
        <div class="flex gap-1.5">
          <dt>Decision threshold</dt>
          <dd class="font-mono tabular-nums text-foreground">{prediction.threshold.toFixed(2)}</dd>
        </div>
      {/if}
      <div class="flex gap-1.5">
        <dt>Latency</dt>
        <dd class="font-mono tabular-nums text-foreground">{formatLatency(prediction.latencyMs)}</dd>
      </div>
    </dl>
    {#if allowUploads || prediction.disclaimer}
      <p class="mt-4 text-[12px] text-foreground-muted">{prediction.disclaimer}</p>
    {/if}
  </section>
</Card.Root>
