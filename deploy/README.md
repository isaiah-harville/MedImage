# Demo site

A website where people pick a scan type and see what a published model makes of an image:
its class probabilities and an attribution map of the pixels that drove the top prediction.
Every page says plainly that these are research models, not medical devices.

It runs at `diagnostics.harville.dev`. Anyone can try the demo images,
which are license-cleared test images from each published run's `examples/`, or upload a
PNG or JPEG of at most 10 MB.

## Data handling

Uploaded images are never stored. This is enforced in code and in the deployment, and the
site's privacy page (`web/src/routes/privacy`) states it to users. Request metadata may be
logged; uploaded content never is.

- **In memory only.** The frontend reads the upload with `request.formData()` and
  forwards it from memory. The API reads the body into memory with a 10 MB cap and parses
  it there; Starlette's multipart parser is set never to spool a part to a temporary file,
  and a test fails if one is. Both pods' `/tmp` is a RAM-backed `emptyDir`, so even a
  library that spooled could not reach a node's disk.
- **Gone after the response.** Nothing keeps the image or the result once the response
  is sent. There is no database, cache, or object store. Responses carry
  `Cache-Control: no-store`.
- **Request logs carry metadata only.** Logging a request's time, path, status, client IP,
  and user agent is allowed. Logging the uploaded image, its file name, the result, or any
  request body is not. Today the API runs with access logs off, the frontend logs nothing
  per request, and Traefik writes no access logs. Pod logs go to Loki and are kept for 14
  days.
- **Cloudflare** proxies all traffic to `diagnostics.harville.dev`, uploads included, and
  sees each request and the client's IP address. The privacy page says so and links
  Cloudflare's privacy policy.
- **Rate limiting** holds client IP addresses in Traefik's memory only.
- **No accounts, cookies, analytics, or third-party scripts.**
- **No training on uploads.** Uploads are never used to train or evaluate models.

A change that keeps any uploaded data, logs request contents, keeps logs longer, or adds a
third-party script must update the privacy page and this file in the same change; see
invariant 9 in `AGENTS.md`.

## Parts

- **`opendiagnostics.serve`**, the inference API (`Dockerfile.serve`). It loads every
  bundled model on CPU at startup and implements the [demo API](api.md). Only the
  frontends can reach it, and it makes no outbound connections.
- **`web/`**, the SvelteKit frontend (`web/Dockerfile`), with the privacy policy at
  `/privacy`. Only its server calls the API; the browser never sees the API's address. See
  `web/README.md`.

### Inference service settings

| Variable | Default | Meaning |
|---|---|---|
| `WEIGHTS_DIR` | required (`/weights` in the image) | folder of bundled models |
| `LISTEN_ADDRESS` | `:8080` | `host:port` to listen on |
| `ATTRIBUTION_STEPS` | `16` | Integrated Gradients steps per explanation |
| `MAX_CONCURRENT` | `2` | predictions that may run at once |

## Run it locally

Download and check the published checkpoints, then start the API:

```bash
uv run python scripts/bundle_models.py weights
WEIGHTS_DIR=weights uv run --extra serve opendiagnostics-serve
```

Then start the frontend against it (`API_URL=http://localhost:8080`); see
[web/README.md](../web/README.md).

## Model weights

The image carries the weights (about 200 MB per model). `scripts/bundle_models.py` takes
each project's latest published run and downloads its checkpoint from the public weights
bucket, checking it against the SHA-256 in `artifacts.yaml`. CI caches the downloads, so a
rebuild fetches only changed checkpoints. Until a run is published, the image workflow
skips the build. Each model page links its weights for download; see
[model weights](../docs/weights.md).

## Deployment

| | |
|---|---|
| Images | `ghcr.io/isaiah-harville/opendiagnostics-serve`, `ghcr.io/isaiah-harville/opendiagnostics-web` |
| Manifests | `k8s/` (Kustomize), applied by Flux from the homelab repo |
| Tags | CI pushes `main`, `sha-<commit>`, and a sortable `<UTC time>-<short sha>` per build. Flux image automation in the homelab repo pins the newest sortable tag; CI never pushes to `main`. |
| Owned by homelab | namespace, the Ingress and its host, TLS, rate and body-size limits, the `ghcr-pull-secret`, the weights bucket and its public route |
