"""Download CAMELYON16 (CC0) from the AWS open data bucket, resumably and MD5-checked.

    uv run python scripts/datasets/camelyon16.py [--out data/camelyon16] [--dry-run]

Training does not need this: ``opendiagnostics prepare`` streams each slide, extracts its
features and deletes it. Use this script to keep the slides on disk, or for the annotations
and masks.

The 399 slides are 752.5 GB; ``--parts annotations masks`` adds the lesion XML (0.08 GB)
and the tissue/tumour masks (9.4 GB). The labels come from ``evaluation/reference.csv``.
"""

from opendiagnostics.data.camelyon_download import download, script_parser

LABELS = "evaluation/reference.csv"


def main() -> None:
    args = script_parser("CAMELYON16", __doc__.split("\n\n")[0]).parse_args()
    folders = tuple(f"{part}/" for part in args.parts)
    download(
        "CAMELYON16",
        args.out,
        lambda name: name == LABELS or name.startswith(folders),
        args.dry_run,
    )


if __name__ == "__main__":
    main()
