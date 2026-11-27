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
  forwards it from memory as base64 in a JSON body. LitServe's size limit refuses a body
  over the cap before it is all read; the API parses the JSON in memory and hands the image
  to its worker over an in-memory queue. A test fails if an upload is spooled to a
  temporary file. Both pods' `/tmp` is a RAM-backed `emptyDir`, so even a library that
  spooled could not reach a node's disk.
- **Gone after the response.** Nothing keeps the image or the result once the response
  is sent. Nothing stores uploads or results: the only volume holds published models
  downloaded from the weights bucket. Responses carry `Cache-Control: no-store`.
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

- **`opendiagnostics.serve`**, the inference API (`Dockerfile.serve`), built on
  [LitServe](https://lightning.ai/docs/litserve). One LitAPI at `POST /predict` runs every
  published model in a worker process; the API process lists models and serves demo images.
  It implements the [demo API](api.md). Only the frontend can
  reach it. Its only outbound connections are DNS and the in-cluster S3 port, used to
  download published models.
- **`web/`**, the SvelteKit frontend (`web/Dockerfile`), with the privacy policy at
  `/privacy`. Only its server calls the API; the browser never sees the API's address. See
  `web/README.md`.

### Inference service settings

| Variable | Default | Meaning |
|---|---|---|
| `WEIGHTS_CACHE` | required | writable folder for downloaded runs and checkpoints |
| `WEIGHTS_BASE_URL` | the public weights URL | where the weights bucket is read from |
| `REFRESH_SECONDS` | `600` | how often to check the catalog for new models; `0` checks only at startup |
| `LISTEN_ADDRESS` | `:8080` | `host:port` to listen on |
| `ATTRIBUTION_STEPS` | `16` | Integrated Gradients steps per explanation |
| `WORKERS` | `1` | LitServe worker processes; each holds every model, so each adds its memory |

## Run it locally

Start the API. It downloads the published models from the public bucket into the cache
folder first; the public route allows 10 requests a minute, so the first start waits a few
minutes:

```bash
WEIGHTS_CACHE=weights-cache uv run --extra serve opendiagnostics-serve
```

Then start the frontend against it (`API_URL=http://localhost:8080`); see
[web/README.md](../web/README.md).

## Models

The image holds no models, so the site deploys whether or not anything is published.
`opendiagnostics publish` puts each published run in the weights bucket and records it in
`catalog.json`:

```
catalog.json                         {"models": {"<project>": "<run-id>", ...}}
<project>/<run-id>/demo.tar          config.yaml, metrics.json, artifacts.yaml, examples/
<project>/<run-id>/checkpoint.ckpt
```

At startup the API reads the catalog from the in-cluster SeaweedFS S3 endpoint. It downloads
each run's `demo.tar` into the cache volume, and its checkpoint as `<sha256>.ckpt` after
checking the size and SHA-256 against `artifacts.yaml`. Anything already cached is not
downloaded again, and cached files the catalog no longer names are deleted. LitServe then
starts one worker that loads every model. A model that fails to download or load is logged
and left out. With no catalog, the API serves no models and the site says so.

LitServe fixes its workers at startup, so new models need a restart. Every
`REFRESH_SECONDS` the API checks the catalog; when it names a different set of runs, the API
downloads them and exits cleanly, and Kubernetes restarts it on the new set. The demo is
down for the restart, about a minute, since the new models are already cached. See [model weights](../docs/weights.md).

## Deployment

| | |
|---|---|
| Images | `ghcr.io/isaiah-harville/opendiagnostics-serve`, `ghcr.io/isaiah-harville/opendiagnostics-web` |
| Manifests | `k8s/` (Kustomize), applied by Flux from the homelab repo |
| Tags | CI pushes `main`, `sha-<commit>`, and a sortable `<UTC time>-<short sha>` per build. Flux image automation in the homelab repo pins the newest sortable tag; CI never pushes to `main`. |
| Owned by homelab | namespace, the Ingress and its host, TLS, rate and body-size limits, the `ghcr-pull-secret`, the weights bucket, its public route, and SeaweedFS's network policy, which must admit the `opendiagnostics` namespace |
