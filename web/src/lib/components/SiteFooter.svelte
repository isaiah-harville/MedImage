<script lang="ts">
import type { DatasetCredit } from '#lib/format.ts';

let { credits }: { credits: DatasetCredit[] } = $props();
</script>

<footer class="mt-16 border-t border-line bg-surface text-sm text-fg-secondary">
  <div class="mx-auto grid max-w-6xl gap-8 px-4 py-10 sm:px-6 md:grid-cols-[2fr_1fr]">
    <section aria-labelledby="footer-datasets">
      <h2 id="footer-datasets" class="font-semibold text-fg">Datasets</h2>
      {#if credits.length === 0}
        <p class="mt-2">Dataset credits appear here once the model service responds.</p>
      {:else}
        <ul class="mt-2 space-y-3">
          {#each credits as credit (credit.dataset)}
            <li>
              <span class="font-medium text-fg">{credit.dataset}</span>, {credit.license}.
              <span class="block">{credit.citation}</span>
            </li>
          {/each}
        </ul>
      {/if}
    </section>
    <section aria-labelledby="footer-project" class="space-y-2">
      <h2 id="footer-project" class="font-semibold text-fg">Research use only</h2>
      <p>Not a medical device. Not for diagnosis or treatment.</p>
      <ul class="space-y-1">
        <li>
          <a
            class="font-medium text-accent underline underline-offset-2 hover:text-accent-hover"
            href="/privacy">Privacy</a
          >
        </li>
        <li>
          <a
            class="font-medium text-accent underline underline-offset-2 hover:text-accent-hover"
            href="https://github.com/isaiah-harville/opendiagnostics">Source code on GitHub</a
          >
        </li>
      </ul>
    </section>
  </div>
</footer>
