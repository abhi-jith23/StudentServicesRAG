from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.collection_config import (  # noqa: E402
    DISCOVERED_LINKS_CSV,
    SOURCES_CSV,
)
from src.source_catalog import (  # noqa: E402
    REQUIRED_COLUMNS,
    load_sources,
    parse_bool,
)


def read_rows(
    path: Path,
) -> list[dict[str, str]]:
    if (
        not path.exists()
        or path.stat().st_size == 0
    ):
        return []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return [
            {
                key: (value or "").strip()
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def write_rows(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise discovered links and "
            "export approved candidates."
        )
    )

    parser.add_argument(
        "--links",
        type=Path,
        default=DISCOVERED_LINKS_CSV,
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=SOURCES_CSV,
    )

    parser.add_argument(
        "--export-approved",
        type=Path,
        default=None,
        help=(
            "Export links marked TRUE to a "
            "sources.csv-compatible template. "
            "Metadata fields that cannot be "
            "inferred remain blank."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.links)

    if not rows:
        print(
            "No discovered links found in "
            f"{args.links}"
        )
        return 0

    approved: list[dict[str, str]] = []
    invalid_boolean_rows: list[int] = []

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        try:
            is_approved = parse_bool(
                row.get("approved", ""),
                field_name="approved",
                row_number=row_number,
            )

        except ValueError:
            invalid_boolean_rows.append(
                row_number
            )
            continue

        if is_approved:
            approved.append(row)

    print(
        f"Total discovered links: {len(rows)}"
    )

    print(
        f"Marked approved: {len(approved)}"
    )

    print(
        "Invalid approved values: "
        f"{len(invalid_boolean_rows)}"
    )

    print("\nBy domain:")

    for domain, count in Counter(
        row.get("domain", "")
        for row in rows
    ).most_common():
        print(
            f"  {domain or '[blank]'}: "
            f"{count}"
        )

    print("\nBy probable type:")

    for source_type, count in Counter(
        row.get("probable_type", "")
        for row in rows
    ).most_common():
        print(
            f"  {source_type or '[blank]'}: "
            f"{count}"
        )

    if invalid_boolean_rows:
        print(
            "\nRows with invalid approved "
            "values: "
            + ", ".join(
                map(
                    str,
                    invalid_boolean_rows,
                )
            )
        )

    if args.export_approved is None:
        print(
            "\nReview "
            "data/catalog/discovered_links.csv "
            "in VS Code. Set approved to TRUE "
            "only for relevant official sources."
        )
        return 0

    existing_sources = load_sources(
        args.catalog,
        approved_only=False,
    )

    existing_urls = {
        source.url
        for source in existing_sources
    }

    export_rows: list[
        dict[str, str]
    ] = []

    for row in approved:
        url = row.get("url", "")

        if (
            not url
            or url in existing_urls
        ):
            continue

        domain = row.get(
            "domain",
            "",
        )

        title = (
            row.get("anchor_text", "")
            or "REPLACE_WITH_TITLE"
        )

        is_external = (
            "FALSE"
            if (
                domain == "uni.lu"
                or domain.endswith(
                    ".uni.lu"
                )
            )
            else "TRUE"
        )

        export_rows.append(
            {
                "source_id": "",
                "title": title,
                "url": url,
                "source_group": "",
                "audience": "",
                "degree_level": "",
                "faculty": "",
                "programme_name": "",
                "language": "en",
                "is_external": is_external,
                "approved": "FALSE",
            }
        )

    write_rows(
        args.export_approved,
        REQUIRED_COLUMNS,
        export_rows,
    )

    print(
        f"\nExported {len(export_rows)} "
        "candidate rows to "
        f"{args.export_approved}"
    )

    print(
        "Complete every blank metadata field, "
        "assign a unique source_id, then copy "
        "reviewed rows into sources.csv and "
        "set approved=TRUE."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())