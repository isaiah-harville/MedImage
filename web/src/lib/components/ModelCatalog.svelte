<script lang="ts">
import { groupByCategory } from '#lib/catalog.ts';
import type { ModelSummary } from '#lib/contract.ts';
import { formatAuroc } from '#lib/format.ts';

let { models }: { models: ModelSummary[] } = $props();

const groups = $derived(groupByCategory(models));

const TASK: Record<ModelSummary['labelMode'], string> = {
  binary: 'Binary',
  multiclass: 'Multiclass',
  multilabel: 'Multilabel',
};
</script>

<div class="grid gap-8">
  {#each groups as group, i (group.category)}
    <section aria-labelledby="category-{i}">
      <h3 id="category-{i}" class="mb-3 text-[15px] font-semibold tracking-[-0.01em]">
        {group.category}
        <span class="ml-1.5 font-mono text-[12px] font-normal text-foreground-muted tabular-nums"
          >{group.models.length}</span
        >
      </h3>
      <div class="overflow-hidden rounded-[var(--radius-xl)] bg-card shadow-[var(--elevation-control)]">
        <table class="w-full text-left text-[13px] md:table-fixed">
          <caption class="sr-only">
            {group.category} models, with their test AUROC and 95% bootstrap interval
          </caption>
          <thead class="hidden border-b border-border text-[12px] text-foreground-muted md:table-header-group">
            <tr>
              <th scope="col" class="w-[32%] px-5 py-3 font-medium">Model</th>
              <th scope="col" class="w-[15%] px-3 py-3 font-medium">Modality</th>
              <th scope="col" class="w-[17%] px-3 py-3 font-medium">Task</th>
              <th scope="col" class="w-[11%] px-3 py-3 text-right font-medium">Test AUROC</th>
              <th scope="col" class="w-[13%] px-3 py-3 text-right font-medium">95% CI</th>
              <th scope="col" class="px-5 py-3 font-medium">Data license</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-border">
            {#each group.models as model (model.id)}
              <tr class="group relative grid grid-cols-[1fr_auto] gap-x-4 gap-y-1 px-5 py-4 hover:bg-secondary/60 md:table-row md:p-0">
                <th scope="row" class="col-span-2 font-normal md:px-5 md:py-3.5">
                  <a
                    href="/models/{model.id}"
                    class="font-medium text-foreground after:absolute after:inset-0 focus-visible:outline-none group-has-[a:focus-visible]:underline"
                    >{model.scanType}</a
                  >
                  <span class="block text-[12px] text-foreground-muted">{model.question}</span>
                </th>
                <td class="text-foreground-muted md:px-3 md:py-3.5">{model.modality}</td>
                <td class="hidden text-foreground-muted md:table-cell md:px-3 md:py-3.5">
                  {TASK[model.labelMode]}, {model.classes.length} classes
                </td>
                <td class="text-right font-mono font-medium tabular-nums md:px-3 md:py-3.5">
                  {formatAuroc(model.testAuroc)}
                </td>
                <td class="hidden text-right font-mono text-[12px] text-foreground-muted tabular-nums md:table-cell md:px-3 md:py-3.5">
                  {model.testAuroc
                    ? `${model.testAuroc.lower.toFixed(3)} to ${model.testAuroc.upper.toFixed(3)}`
                    : ''}
                </td>
                <td class="hidden text-foreground-muted md:table-cell md:px-5 md:py-3.5">{model.license}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>
  {/each}
</div>
