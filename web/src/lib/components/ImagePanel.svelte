<script lang="ts">
let { src, alt, caption }: { src: string; alt: string; caption: string } = $props();
</script>

<figure class="min-w-0">
  <img
    {src}
    {alt}
    width="224"
    height="224"
    class="aspect-square w-full rounded-xl bg-muted object-contain [image-rendering:pixelated]"
  />
  <figcaption class="mt-2 text-sm text-fg-secondary">{caption}</figcaption>
</figure>
