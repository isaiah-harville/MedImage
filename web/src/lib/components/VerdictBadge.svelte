<script lang="ts">
import { Badge } from '@sivir-ui/svelte/components/badge';
import type { Verdict } from '#lib/format.ts';

let { verdict, truth }: { verdict: Exclude<Verdict, 'unknown'>; truth: string } = $props();
</script>

<Badge variant={verdict === 'correct' ? 'success' : 'error'} class="w-fit">
  {verdict === 'correct' ? 'Agrees with' : 'Differs from'} reference label: {truth}
</Badge>
