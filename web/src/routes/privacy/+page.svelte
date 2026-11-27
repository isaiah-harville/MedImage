<script lang="ts">
import PolicySection from '#lib/components/PolicySection.svelte';
</script>

<svelte:head>
  <title>Privacy | OpenDiagnostics demo</title>
  <meta name="description" content="How the OpenDiagnostics demo handles images you upload." />
</svelte:head>

<article class="max-w-2xl">
  <h1 class="text-3xl font-semibold tracking-tight">Privacy</h1>
  <p class="mt-2 text-sm text-foreground-muted">Last updated <time datetime="2026-11-27">2026-11-27</time></p>
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
    <p>
      We accept PNG and JPEG images, and for 3D models NumPy (.npy) and NIfTI (.nii, .nii.gz)
      volumes.
    </p>
    <p>
      Do not upload images with names, dates of birth, record numbers, or other identifiers, or
      images you do not have the right to share.
    </p>
    <p>
      The model reads only pixels or voxels. We do not store metadata (EXIF or NIfTI headers),
      and the images we send back are re-encoded without metadata. Remove names and dates from a
      NIfTI header before you upload it.
    </p>
  </PolicySection>

  <PolicySection title="Tracking">
    <p>No accounts, no cookies, no analytics, no trackers, and no third-party scripts or fonts.</p>
  </PolicySection>

  <PolicySection title="Request logs">
    <p>
      To run and protect the site, our servers may log details of each request: the time, the page
      or file asked for, the response status, your IP address, and your browser's user agent. These
      logs never contain the images you upload, their file names, or the results, and they are kept
      for 14 days.
    </p>
    <p>
      Rate limits are enforced on IP addresses, which the reverse proxy holds in memory only
      briefly.
    </p>
  </PolicySection>

  <PolicySection title="Cloudflare">
    <p>
      All traffic to this site, including uploads, passes through Cloudflare on its way to our
      servers. Cloudflare sees each request, including your IP address, and handles it under the
      <a
        class="font-medium text-primary underline underline-offset-2 hover:opacity-80"
        href="https://www.cloudflare.com/privacypolicy/">Cloudflare privacy policy</a
      >.
    </p>
  </PolicySection>

  <PolicySection title="Model weights">
    <p>
      Downloads of model weights are handled like any other request: they may be logged as above,
      and nothing else about you is kept.
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
        class="font-medium text-primary underline underline-offset-2 hover:opacity-80"
        href="https://github.com/isaiah-harville/opendiagnostics/issues"
        >github.com/isaiah-harville/opendiagnostics/issues</a
      >.
    </p>
  </PolicySection>
</article>
