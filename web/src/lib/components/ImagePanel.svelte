<script lang="ts">
let { src, alt, caption }: { src: string; alt: string; caption: string } = $props();
</script>

<figure class="min-w-0">
  <figcaption class="mb-2 font-mono text-[11px] tracking-wide text-[var(--color-viewer-foreground)] uppercase">
    {caption}
  </figcaption>
  <img
    {src}
    {alt}
    width="224"
    height="224"
    class="aspect-square w-full bg-black object-contain [image-rendering:pixelated]"
  />
</figure>
