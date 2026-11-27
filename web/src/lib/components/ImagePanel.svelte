<script lang="ts">
let { src, alt, caption }: { src: string; alt: string; caption: string } = $props();
</script>

<figure class="min-w-0">
  <img
    {src}
    {alt}
    width="224"
    height="224"
    class="aspect-square w-full rounded-[var(--radius-lg)] bg-secondary object-contain shadow-[var(--elevation-1)] [image-rendering:pixelated]"
  />
  <figcaption class="mt-2 text-[12px] text-foreground-muted">{caption}</figcaption>
</figure>
