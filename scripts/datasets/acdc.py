"""Download ACDC (cardiac cine MRI: RV, myocardium, LV) and write its manifest.

    uv run python scripts/datasets/acdc.py data/acdc

Files come one by one from the official Girder collection, which is public and needs no
account. Only the end-diastole and end-systole frames, their masks and ``Info.cfg`` are
fetched (176 MB for 150 patients); the 2.3 GB of 4D cine volumes are skipped. Girder
publishes no checksums, so each file is checked against its listed size only. There are no
archives, so ``--delete-archives`` does nothing here.

The official split is kept: the 50 testing patients, whose masks are public, are the test
split, and the 100 training patients are split 80/20 into train and validation, stratified
by diagnosis group. Each patient gives two cases (ED and ES), which stay in one split.

License: CC BY-NC-SA 4.0, for non-commercial scientific research, citing Bernard et al.
2018 (LICENSE_TERMS.md in the collection).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from _download import arguments, assign_splits, download, write_manifest

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
COLLECTION = "637218c173e9f0047faa00fb"
COLUMNS = ("id", "image", "label", "patient_id", "phase", "group", "split")


def get(path: str, **query: str) -> Any:
    url = f"{API}/{path}?{urllib.parse.urlencode(query | {'limit': '0'})}"
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def child(parent_type: str, parent: str, name: str) -> str:
    folders = get("folder", parentType=parent_type, parentId=parent)
    (match,) = [f["_id"] for f in folders if f["name"] == name]
    return match


def info(text: str) -> dict[str, str]:
    pairs = (line.split(":", 1) for line in text.splitlines() if ":" in line)
    return {k.strip(): v.strip() for k, v in pairs}


def main() -> None:
    root = arguments(__doc__.split("\n\n")[0]).root
    database = child("collection", COLLECTION, "database")
    rows = []
    for folder, official in (("training", ""), ("testing", "test")):
        for patient in get(
            "folder", parentType="folder", parentId=child("folder", database, folder)
        ):
            name = patient["name"]
            items = {i["name"]: i for i in get("item", folderId=patient["_id"])}
            for filename, item in items.items():
                if filename == "Info.cfg" or "_frame" in filename:
                    download(
                        f"{API}/item/{item['_id']}/download",
                        root / folder / name / filename,
                        size=item["size"],
                    )
            details = info((root / folder / name / "Info.cfg").read_text())
            for phase in ("ED", "ES"):
                frame = f"{name}_frame{int(details[phase]):02d}"
                rows.append(
                    {
                        "id": frame,
                        "image": f"{folder}/{name}/{frame}.nii.gz",
                        "label": f"{folder}/{name}/{frame}_gt.nii.gz",
                        "patient_id": name,
                        "phase": phase,
                        "group": details["Group"],
                        "split": official,
                    }
                )
    training = [r for r in rows if not r["split"]]
    splits = assign_splits(
        [r["patient_id"] for r in training], [r["group"] for r in training], (0.8, 0.2, 0.0)
    )
    for row, split in zip(training, splits, strict=True):
        row["split"] = split
    write_manifest(root, rows, COLUMNS)


if __name__ == "__main__":
    main()
