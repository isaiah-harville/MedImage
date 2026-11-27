# Demo API

The contract between the demo frontend (`web/`) and the inference service
(`opendiagnostics.serve`, built on LitServe). All responses are JSON unless stated. The
service runs on CPU and holds no state: uploads are processed in memory and never stored or
logged, and every response carries `Cache-Control: no-store`.

## `GET /health`

LitServe's health check: `200 ok` once its worker has loaded every model, `503` before.

## `GET /models`

One entry per servable model, in display order:

```json
[
  {
    "id": "pneumoniamnist-convnext",
    "scan_type": "Pediatric chest X-ray",
    "modality": "X-ray",
    "category": "X-ray and mammography",
    "question": "Pneumonia or normal?",
    "label_mode": "binary",
    "classes": ["normal", "pneumonia"],
    "input": {"channels": 1, "size": 224, "dims": 2},
    "dataset": "PneumoniaMNIST (MedMNIST v2)",
    "license": "CC BY 4.0",
    "citation": "Yang, J. et al. ...",
    "run_id": "20261127-031500-abcd1234",
    "test_auroc": {"value": 0.97, "lower": 0.96, "upper": 0.98},
    "demo_count": 8,
    "weights": {
      "url": "https://diagnostics.harville.dev/opendiagnostics-weights/pneumoniamnist-convnext/20261127-031500-abcd1234/checkpoint.ckpt",
      "bytes": 200123456,
      "sha256": "9f2c..."
    }
  }
]
```

Models come sorted by `category`, the group of imaging types the site lists them under
(CT and MRI, X-ray and mammography, Ultrasound, Pathology and microscopy, Skin, Eye, Other).
`test_auroc` is `null` when the run has no test metrics. `weights` is the public download of
the served checkpoint, and `null` when it was not published (see [model weights](../docs/weights.md)).

## `GET /models/{id}/demo`

Curated test images from the open dataset, with ground truth:

```json
[{"sample_id": "test-12", "truth": "pneumonia", "image_url": "/models/pneumoniamnist-convnext/demo/test-12.png"}]
```

## `GET /models/{id}/demo/{sample_id}.png`

The demo image as `image/png`.

## `POST /predict`

Runs a model on one image or volume. The JSON body names the model and gives exactly one
input:

```json
{"model": "pneumoniamnist-convnext", "sample_id": "test-12"}
{"model": "pneumoniamnist-convnext", "image": "<base64 PNG or JPEG>"}
{"model": "synapsemnist3d-resnet18", "image": "<base64 .npy or NIfTI volume>"}
```

`sample_id` runs a demo image; `image` runs an upload of at most 10 MB. A 2D model
(`input.dims` 2) takes a PNG or JPEG of at most 4096 x 4096 pixels. A 3D model takes a
`.npy` array or a NIfTI-1 or NIfTI-2 file, either one optionally gzipped, holding one 3D
volume of at most 1024 voxels a side that unpacks to at most 256 MB. NIfTI axes are read as
(x, y, z) and reversed, so slices run along z. A `uint8` volume is divided by 255; any other
is stretched from its minimum to its maximum. The volume is then resized to the model's
cube. Returns a `Prediction`. Errors:

| Status | When |
|---|---|
| 404 | unknown model id or sample id |
| 413 | body over about 13.4 MB, file over 10 MB, or image over 4096 x 4096 |
| 415 | a 2D model sent something other than a PNG or JPEG |
| 422 | neither or both inputs, `image` not base64, or the file could not be decoded (for a 3D model, the detail says why) |

## `Prediction`

```json
{
  "model": "pneumoniamnist-convnext",
  "label_mode": "binary",
  "probabilities": [{"class": "normal", "probability": 0.08}, {"class": "pneumonia", "probability": 0.92}],
  "predicted": ["pneumonia"],
  "threshold": 0.41,
  "input_png": "data:image/png;base64,...",
  "attribution_png": "data:image/png;base64,...",
  "slices": null,
  "truth": "pneumonia",
  "disclaimer": "Research model, not a medical device. ...",
  "latency_ms": 840
}
```

- `probabilities` lists every class, highest first. Binary models give both classes.
- `predicted` is the predicted class (multiclass), the class at or above `threshold`
  (binary; `threshold` was fitted on validation data), or every label at or above 0.5
  (multilabel; `threshold` is 0.5).
- `input_png` is the image as the model saw it (224 x 224). `attribution_png` is that image
  with an Integrated Gradients map of the pixels that drove the top prediction. For a 3D
  model they show the middle slice.
- `slices` is `null` for a 2D model. For a 3D model it holds `input` and `attribution`, one
  PNG per slice along the first axis, at the model's resolution.
- `truth` is set for demo images and `null` for uploads.
