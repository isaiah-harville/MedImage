# Demo site

A website where people pick a scan type and see what a published model makes of an image:
its class probabilities and an attribution map of the pixels that drove the top prediction.
Every page says plainly that these are research models, not medical devices.

There are two deployments of one frontend:

| Site | Uploads | Exposure |
|---|---|---|
| Public | Off. Demo images only: license-cleared test images from each published run's `examples/`. | `traefik-public` |
| Internal | On. PNG or JPEG, at most 10 MB, processed in memory and never stored or logged. | `traefik-internal`, behind Authentik |

## Parts

- **`opendiagnostics.serve`**, the inference API (`Dockerfile.serve`). It loads every
  bundled model on CPU at startup and implements the [demo API](demo-api.md). Only the
  frontends can reach it, and it makes no outbound connections.
- **`web/`**, the SvelteKit frontend (`web/Dockerfile`). Only its server calls the API; the
  browser never sees the API's address. See `web/README.md`.

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
| Owned by homelab | namespace, the two Ingresses and their hosts, TLS, Authentik on the internal host, rate limits, the `ghcr-pull-secret` |
