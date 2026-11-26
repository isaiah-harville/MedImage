<script lang="ts">
import PolicySection from '#lib/components/PolicySection.svelte';
</script>

<svelte:head>
  <title>Privacy | OpenDiagnostics demo</title>
  <meta name="description" content="How the OpenDiagnostics demo handles images you upload." />
</svelte:head>

<article class="max-w-2xl">
  <h1 class="text-3xl font-semibold tracking-tight">Privacy</h1>
  <p class="mt-2 text-sm text-fg-secondary">Last updated <time datetime="2026-11-26">2026-11-26</time></p>
  <p class="mt-4 text-lg">
    This site runs research models on medical images. Here is what happens to an image you upload
    and what we keep.
  </p>

  <PolicySection title="Images you upload">
    <p>
      Uploaded images are processed only in server memory for the length of the request, to compute
      the prediction. They are <strong>never written to disk, stored, logged, or used to train
      models</strong>, and they are gone when the response is sent.
    </p>
    <p>
      The result (the probabilities and the attribution image) goes back to your browser and is not
      kept on the server.
    </p>
  </PolicySection>

  <PolicySection title="What you may upload">
    <p>We accept only PNG and JPEG images.</p>
    <p>
      Do not upload images with names, dates of birth, record numbers, or other identifiers, or
      images you do not have the right to share.
    </p>
    <p>
      The model reads only pixels. We do not store metadata (EXIF), and the image we send back is
      re-encoded without metadata.
    </p>
  </PolicySection>

  <PolicySection title="Tracking">
    <p>No accounts, no cookies, no analytics, no trackers, and no third-party scripts or fonts.</p>
  </PolicySection>

  <PolicySection title="Server operations">
    <p>The web server and the inference service write no per-request logs.</p>
    <p>
      Application error logs, which never contain images or IP addresses, are kept for 14 days.
    </p>
    <p>
      To enforce rate limits, the reverse proxy holds client IP addresses in memory only briefly.
      It does not write access logs.
    </p>
  </PolicySection>

  <PolicySection title="Demo images">
    <p>
      Demo images come from open research datasets (MedMNIST). Each dataset's license is shown on
      its model page.
    </p>
  </PolicySection>

  <PolicySection title="Not medical advice">
    <p>
      The models are research software, not a medical device. Do not use the results for any
      medical decision.
    </p>
  </PolicySection>

  <PolicySection title="Contact">
    <p>
      Open an issue at
      <a
        class="font-medium text-accent underline underline-offset-2 hover:text-accent-hover"
        href="https://github.com/isaiah-harville/opendiagnostics/issues"
        >github.com/isaiah-harville/opendiagnostics/issues</a
      >.
    </p>
  </PolicySection>
</article>
