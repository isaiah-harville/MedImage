<script lang="ts">
import { Badge } from '@sivir-ui/svelte/components/badge';
import type { Verdict } from '#lib/format.ts';

let { verdict, truth }: { verdict: Exclude<Verdict, 'unknown'>; truth: string } = $props();
</script>

<Badge variant={verdict === 'correct' ? 'success' : 'error'} dot class="w-fit">
  {verdict === 'correct' ? 'Correct' : 'Wrong'}: the dataset labels it {truth}
</Badge>
