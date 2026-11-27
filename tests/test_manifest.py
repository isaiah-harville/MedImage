import csv

import numpy as np
from PIL import Image

from opendiagnostics.data.manifest import CSVImageDataModule


def test_images_come_out_channels_first_with_height_before_width(tmp_path):
    rows = []
    for i in range(6):
        pixels = np.zeros((6, 4, 3), np.uint8)
        pixels[0, :, 0] = 255
        Image.fromarray(pixels).save(tmp_path / f"{i}.png")
        split = ["train", "val", "test"][i % 3]
        rows.append(
            {"path": f"{i}.png", "patient_id": f"p{i}", "label": "ab"[i % 2], "split": split}
        )
    with (tmp_path / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data = CSVImageDataModule(
        manifest=str(tmp_path / "manifest.csv"), split_column="split", in_channels=3, batch_size=2
    )
    data.setup()
    image = next(iter(data.train_dataloader()))["image"][0]
    assert tuple(image.shape) == (3, 6, 4)
    assert image[0, 0].min() > 0.99
    assert image[0, 1:].max() == 0
