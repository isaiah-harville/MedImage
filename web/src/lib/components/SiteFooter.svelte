<script lang="ts">
import type { DatasetCredit } from '#lib/format.ts';

let { credits }: { credits: DatasetCredit[] } = $props();
</script>

<footer class="mt-24 border-t border-border bg-card text-[13px] text-foreground-muted">
  <div class="mx-auto grid max-w-6xl gap-10 px-5 py-12 md:grid-cols-[2fr_1fr]">
    <section aria-labelledby="footer-datasets">
      <h2 id="footer-datasets" class="font-semibold text-foreground">Datasets</h2>
      {#if credits.length === 0}
        <p class="mt-3">Dataset credits appear here once the model service responds.</p>
      {:else}
        <ul class="mt-3 grid gap-x-8 gap-y-3 sm:grid-cols-2">
          {#each credits as credit (credit.dataset)}
            <li>
              <span class="font-medium text-foreground">{credit.dataset}</span>
              <span class="ml-1">{credit.license}</span>
              <span class="mt-0.5 block line-clamp-2">{credit.citation}</span>
            </li>
          {/each}
        </ul>
      {/if}
    </section>
    <section aria-labelledby="footer-project" class="space-y-3">
      <h2 id="footer-project" class="font-semibold text-foreground">OpenDiagnostics</h2>
      <p>Open research models. Not a medical device, and not for diagnosis or treatment.</p>
      <ul class="space-y-1.5">
        <li><a class="text-foreground underline-offset-4 hover:underline" href="/privacy">Privacy</a></li>
        <li>
          <a class="text-foreground underline-offset-4 hover:underline" href="https://github.com/isaiah-harville/opendiagnostics"
            >Source code and published runs</a
          >
        </li>
      </ul>
    </section>
  </div>
</footer>
