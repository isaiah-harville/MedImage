<script lang="ts">
import type { PageData } from './$types';

let { data }: { data: PageData } = $props();
</script>

<svelte:head>
  <title>Papers | OpenDiagnostics</title>
  <meta name="description" content="Conference drafts on the published OpenDiagnostics models." />
</svelte:head>

<header class="border-b border-border pb-8">
  <h1 class="text-[28px] leading-tight font-semibold tracking-[-0.02em]">Papers</h1>
  <p class="mt-2 max-w-2xl text-[14px] leading-relaxed text-foreground-muted">
    Drafts in IEEE conference format on the models whose test AUROC interval lies above every
    published result we could source. Each reports one training run, and every number in it comes
    from that run's public files.
  </p>
</header>

{#if data.papers.length === 0}
  <p class="mt-8 rounded-[var(--radius-lg)] bg-secondary px-4 py-6 text-foreground-muted">
    No papers yet. A paper appears here once its draft is added to the repository's papers folder.
  </p>
{:else}
  <ul class="divide-y divide-border">
    {#each data.papers as paper (paper.name)}
      <li class="grid gap-6 py-8 sm:grid-cols-[minmax(0,1fr)_10rem] sm:gap-10">
        <article>
          <h2 class="text-[18px] leading-snug font-semibold tracking-[-0.01em] text-balance">
            {paper.title}
          </h2>
          <p class="mt-2 text-[14px] leading-relaxed text-foreground-muted">{paper.description}</p>
          <p class="mt-3 text-[13px]">
            <a href="/models/{paper.project}" class="text-primary underline-offset-4 hover:underline"
              >Try the model</a
            >
            <span class="mx-2 text-border-strong" aria-hidden="true">|</span>
            <a
              href="/papers/{paper.name}/paper.pdf"
              target="_blank"
              rel="noopener"
              class="text-primary underline-offset-4 hover:underline">Read the PDF</a
            >
          </p>
        </article>
        <a
          href="/papers/{paper.name}/paper.pdf"
          target="_blank"
          rel="noopener"
          aria-label="Open {paper.title} as a PDF in a new tab"
          class="group block w-40 justify-self-start sm:justify-self-end"
        >
          <img
            src="/papers/{paper.name}/preview.png"
            alt=""
            width="320"
            height="414"
            loading="lazy"
            class="w-full rounded-[var(--radius-md)] bg-white shadow-[var(--elevation-control)] transition-[box-shadow,translate] duration-200 ease-[var(--ease-out)] group-hover:-translate-y-0.5 group-hover:shadow-[var(--elevation-float)]"
          />
          <span class="mt-2 block text-center text-[12px] text-foreground-muted group-hover:text-foreground">
            Open PDF
          </span>
        </a>
      </li>
    {/each}
  </ul>
{/if}
