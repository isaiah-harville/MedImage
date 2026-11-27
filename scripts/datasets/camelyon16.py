"""Download CAMELYON16 (CC0) from the AWS open data bucket, resumably and MD5-checked.

    uv run python scripts/datasets/camelyon16.py [--out data/camelyon16] [--dry-run]

The 399 slides are 752.5 GB; ``--parts annotations masks`` adds the lesion XML (0.08 GB)
and the tissue/tumour masks (9.4 GB). The labels come from ``evaluation/reference.csv``.
"""

from camelyon_s3 import download, parser

LABELS = "evaluation/reference.csv"


def main() -> None:
    args = parser("CAMELYON16", __doc__.split("\n\n")[0]).parse_args()
    folders = tuple(f"{part}/" for part in args.parts)
    download(
        "CAMELYON16",
        args.out,
        lambda name: name == LABELS or name.startswith(folders),
        args.dry_run,
    )


if __name__ == "__main__":
    main()
