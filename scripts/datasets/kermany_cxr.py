"""Download the Kermany pediatric chest X-ray set (normal, pneumonia) and write manifest.csv.

    uv run python scripts/datasets/kermany_cxr.py data/kermany_cxr [--archive PATH] [--delete-archives]

Downloads ZhangLabData.zip (8.44 GB, SHA-256 checked) from Mendeley Data and unpacks only
CellData/chest_xray: 5,232 training and 624 test JPEGs. Bacterial and viral pneumonia share
the PNEUMONIA label. The patient ID is the number in each filename. Pass the same --archive
to kermany_oct.py to download the zip once.
"""

from __future__ import annotations

from _kermany import main

LABELS = ("NORMAL", "PNEUMONIA")

if __name__ == "__main__":
    main(__doc__, "chest_xray", LABELS)
