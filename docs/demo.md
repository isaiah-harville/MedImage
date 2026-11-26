# Demo site

A website where people pick a scan type and see what a published model makes of an image:
its class probabilities and an attribution map of the pixels that drove the top prediction.
Every page says plainly that these are research models, not medical devices.

It runs at `diagnostics.harville.dev` on `traefik-public`. Anyone can try the demo images,
which are license-cleared test images from each published run's `examples/`, or upload a
PNG or JPEG of at most 10 MB.

## Data handling

Uploaded images are never stored. This is enforced in code and in the deployment, and the
site's privacy page (`web/src/routes/privacy`) states it to users.

- **In memory only.** The frontend reads the upload with `request.formData()` and
  forwards it from memory. The API reads the body into memory with a 10 MB cap and parses
  it there; Starlette's multipart parser is set never to spool a part to a temporary file,
  and a test fails if one is. Both pods' `/tmp` is a RAM-backed `emptyDir`, so even a
  library that spooled could not reach a node's disk.
- **Gone after the response.** Nothing keeps the image or the result once the response
  is sent. There is no database, cache, or object store. Responses carry
  `Cache-Control: no-store`.
- **No logs of requests.** The API runs with access logs off and logs nothing per request.
  The frontend logs nothing about requests either. Traefik writes no access logs. Pod
  logs (errors only, never images or IP addresses) go to Loki and are kept for 14 days.
- **Rate limiting** holds client IP addresses in Traefik's memory only.
- **No accounts, cookies, analytics, or third-party scripts.**
- **No training on uploads.** Uploads are never used to train or evaluate models.

A change that keeps any uploaded data, adds request logging, or adds a third-party script
must update the privacy page in the same change; see invariant 9 in `AGENTS.md`.

## Parts

- **`opendiagnostics.serve`**, the inference API (`Dockerfile.serve`). It loads every
  bundled model on CPU at startup and implements the [demo API](demo-api.md). Only the
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

Bundle the published models, with checkpoints from your local `runs/` or the URLs in each
run's `artifacts.yaml`, then start the API:

```bash
uv run python scripts/bundle_models.py weights
WEIGHTS_DIR=weights uv run --extra serve opendiagnostics-serve
```

Then start the frontend against it (`API_URL=http://localhost:8080`); see `web/README.md`.

## Model weights

The image carries the weights (about 200 MB per model). `scripts/bundle_models.py` takes
each project's latest published run and checks its checkpoint against the SHA-256 in
`artifacts.yaml`. CI downloads checkpoints from the `url` in that file, so a run is
servable once its checkpoint is uploaded somewhere CI can reach and the URL is recorded.
Until then the image workflow skips the build. Weights trained on CC BY-NC data
(DermaMNIST) may be used only non-commercially.

## Deployment

| | |
|---|---|
| Images | `ghcr.io/isaiah-harville/opendiagnostics-serve`, `ghcr.io/isaiah-harville/opendiagnostics-web` |
| Manifests | `deploy/k8s/` (Kustomize), applied by Flux from the homelab repo |
| Tags | CI pins `newTag` in `deploy/k8s/kustomization.yaml` to `sha-<commit>` on each build |
| Owned by homelab | namespace, the Ingress and its host, TLS, rate and body-size limits, the `ghcr-pull-secret` |
