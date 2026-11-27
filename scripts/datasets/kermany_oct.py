"""Download the Kermany retinal OCT set (CNV, DME, drusen, normal) and write manifest.csv.

    uv run python scripts/datasets/kermany_oct.py data/kermany_oct [--archive PATH] [--delete-archives]

Downloads ZhangLabData.zip (8.44 GB, SHA-256 checked) from Mendeley Data and unpacks only
CellData/OCT: 108,309 training and 1,000 test JPEGs. The patient ID is the number in each
filename. Pass the same --archive to kermany_cxr.py to download the zip once.
"""

from __future__ import annotations

from _kermany import main

LABELS = ("CNV", "DME", "DRUSEN", "NORMAL")

if __name__ == "__main__":
    main(__doc__, "OCT", LABELS)
