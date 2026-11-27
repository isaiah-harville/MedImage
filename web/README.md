# OpenDiagnostics demo site

A SvelteKit site where people pick a scan type, run an OpenDiagnostics research model on an image,
and see the prediction with an attribution map. These are research models, not a medical device,
and every page says so.

The public site (diagnostics.harville.dev) shows demo scans from each model's open test set and lets
anyone upload their own PNG or JPEG, after ticking a consent box. Uploads are processed in memory
and never stored; see `/privacy`.

`ALLOW_UPLOADS=false` is an off switch for uploads, for example during an incident: the upload form
disappears and the upload action returns 404. Production leaves uploads on.

Only the SvelteKit server talks to the inference service, through page loads, form actions, and an
image proxy. The browser never sees `API_URL`. The contract between the two is
[deploy/api.md](../deploy/api.md).

## Data handling

`/privacy` states the policy to users. In this server:

- An upload is read with `request.formData()` and forwarded to the inference service from memory.
  It is never written to disk.
- Nothing logs the file, its name, the result, or request contents. Request metadata (time,
  path, status, client IP, user agent) may be logged; today the only logs are
  inference-service errors (name and message).
- Every non-GET response (every form action, so every prediction) has `Cache-Control: no-store`.
  Every response has `Referrer-Policy: same-origin`: no referrer reaches another site.
  (`no-referrer` would make browsers send `Origin: null` on the site's own form posts,
  and SvelteKit rejects those as cross-site.)
- Results travel only in the form action's response body, never in a URL, and the server keeps no
  copy.
- No cookies, analytics, trackers, or third-party scripts. The two fonts (Inter and
  JetBrains Mono) come with Sivir UI and are served from this site, not a font host.

## Design

Components and theme come from [Sivir UI](https://github.com/aidan-neel/sivir-ui)
(`@sivir-ui/svelte`), imported per component (`@sivir-ui/svelte/components/<name>`) so the
build skips the parts the site does not use. `src/app.css` imports its theme, and an inline
script in `src/app.html` sets the `dark` class from the system setting before first paint.
Sivir declares a peer dependency on SvelteKit 2 but uses no SvelteKit API, so
`package.json` overrides that peer to the SvelteKit 3 this app runs.

## Configuration

`src/env.ts` declares the variables, and SvelteKit validates them with the Zod schemas in
`src/lib/server/config.ts` when the server starts. A bad value stops the server with a message
that says what to fix.

| Variable          | Required      | Default                                 | Meaning                                                                                                                   |
| ----------------- | ------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `API_URL`         | yes           |                                         | Base URL of the inference service, for example `http://localhost:8000`.                                                   |
| `ALLOW_UPLOADS`   | no            | `true`                                  | `false` hides the upload form and makes the upload action return 404.                                                     |
| `PORT`            | no            | `3000`                                  | Port the server listens on (adapter-node).                                                                                |
| `ORIGIN`          | in production, unless the two below are set |                                         | Public URL of the site, for example `https://demo.example.org`. SvelteKit rejects form posts whose origin does not match. |
| `PROTOCOL_HEADER`, `HOST_HEADER` | instead of `ORIGIN` |                     | Take the origin from each request, for a site served on several hosts behind a trusted proxy: `x-forwarded-proto` and `host`. |
| `BODY_SIZE_LIMIT` | no            | `11M` in the image                      | Largest request body. Keep it above the 10 MB upload limit.                                                               |
| `PAPERS_DIR`      | no            | `../papers`; `/app/papers` in the image | Folder of papers shown on `/papers`: one subfolder each with `paper.yaml`, the PDF, and `preview.png`.                    |

## Run locally

Start the inference service (or the mock below), then:

```bash
npm ci
API_URL=http://localhost:8000 npm run dev
```

Add `ALLOW_UPLOADS=false` to try the read-only mode.

Without the Python service, `npm run mock-api` serves the contract's example JSON and generated
images on port 8787:

```bash
npm run mock-api &
API_URL=http://localhost:8787 npm run dev
```

## Check

```bash
npm ci && npm run check
```

`check` runs Biome, `svelte-check`, and the Vitest tests. Production build:

```bash
API_URL=http://build.invalid npm run build
API_URL=http://localhost:8000 ORIGIN=http://localhost:3000 node build
```

SvelteKit validates `API_URL` while it analyses the build, so the build needs a placeholder. The
real value is read when the server starts.

## Container

```bash
docker build --build-context papers=../papers -t opendiagnostics-web .
docker run -p 3000:3000 -e API_URL=http://inference:8000 -e ORIGIN=http://localhost:3000 opendiagnostics-web
```

The build stage runs `npm run check` before `vite build`. The runtime image runs as UID 10001 and
serves `GET /healthz`.
