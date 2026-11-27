"""Download CAMELYON17 (CC0) from the AWS open data bucket, resumably and MD5-checked.

    uv run python scripts/datasets/camelyon17.py [--out data/camelyon17] [--dry-run]

Training does not need this: ``opendiagnostics prepare`` streams each slide, extracts its
features and deletes it. Use this script to keep the slides on disk, or for the annotations
and masks.

By default only the 500 slides of the 100 labelled training patients (000-099) come down,
because ``stages.csv`` holds no labels for the 500 test slides. ``--test-slides`` adds them.
"""

import re

from opendiagnostics.data.camelyon_download import download, script_parser

LABELS = "stages.csv"
TRAINING_PATIENT = re.compile(r"patient_0\d\d_")


def main() -> None:
    arguments = script_parser("CAMELYON17", __doc__.split("\n\n")[0])
    arguments.add_argument(
        "--test-slides", action="store_true", help="also download the 500 unlabelled test slides"
    )
    args = arguments.parse_args()
    folders = tuple(f"{part}/" for part in args.parts)

    def wanted(name: str) -> bool:
        if name == LABELS:
            return True
        if not name.startswith(folders):
            return False
        return args.test_slides or bool(TRAINING_PATIENT.search(name))

    download("CAMELYON17", args.out, wanted, args.dry_run)


if __name__ == "__main__":
    main()
