<script lang="ts">
import { Badge } from '@sivir-ui/svelte/components/badge';
</script>

<header class="sticky top-0 z-20 border-b border-border bg-background/85 backdrop-blur">
  <div class="mx-auto flex h-16 max-w-6xl items-center justify-between gap-4 px-5">
    <a href="/" class="flex items-center gap-2.5 font-semibold tracking-[-0.015em] text-foreground">
      <img src="/favicon.svg" alt="" class="size-7" />
      OpenDiagnostics
    </a>
    <nav aria-label="Site" class="flex items-center gap-4 text-[13px] font-medium">
      <a href="/#models" class="hidden text-foreground-muted transition-colors hover:text-foreground sm:inline">
        Models
      </a>
      <a href="/privacy" class="hidden text-foreground-muted transition-colors hover:text-foreground sm:inline">
        Privacy
      </a>
      <Badge variant="warning" dot>Research use only</Badge>
    </nav>
  </div>
</header>
