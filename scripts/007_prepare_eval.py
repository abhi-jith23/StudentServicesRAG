from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.collection_config import (  # noqa: E402
    SOURCES_CSV,
)

from src.config import (  # noqa: E402
    EVAL_DIR,
)

from src.eval import (  # noqa: E402
    EVAL_COLUMNS,
    load_eval_questions,
)

from src.source_catalog import (  # noqa: E402
    load_sources,
)


EVAL_FULL = (
    EVAL_DIR
    / "eval_full.csv"
)

EVAL_DEV = (
    EVAL_DIR
    / "eval_dev.csv"
)

EVAL_HOLDOUT = (
    EVAL_DIR
    / "eval_holdout.csv"
)

SOURCE_REFERENCE = (
    EVAL_DIR
    / "source_reference.csv"
)

SOURCE_REFERENCE_FIELDS = (
    "source_id",
    "title",
    "source_group",
    "audience",
    "degree_level",
    "faculty",
    "programme_name",
    "language",
    "is_external",
    "url",
)


def parse_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the retrieval-"
            "evaluation template and "
            "source reference, or split "
            "a completed eval_full.csv "
            "into dev and holdout files."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--expected-dev",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--expected-holdout",
        type=int,
        default=15,
    )

    return parser.parse_args()


def write_header_only(
    path: Path,
    *,
    overwrite: bool,
) -> None:
    if (
        path.exists()
        and not overwrite
    ):
        return

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
            fieldnames=EVAL_COLUMNS,
        )

        writer.writeheader()


def write_source_reference(
    path: Path,
    *,
    overwrite: bool,
) -> None:
    if (
        path.exists()
        and not overwrite
    ):
        return

    sources = load_sources(
        SOURCES_CSV,
        approved_only=True,
    )

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
            fieldnames=(
                SOURCE_REFERENCE_FIELDS
            ),
        )

        writer.writeheader()

        for source in sources:
            writer.writerow(
                {
                    "source_id": (
                        source.source_id
                    ),
                    "title": (
                        source.title
                    ),
                    "source_group": (
                        source.source_group
                    ),
                    "audience": (
                        source.audience
                    ),
                    "degree_level": (
                        source.degree_level
                    ),
                    "faculty": (
                        source.faculty
                    ),
                    "programme_name": (
                        source.programme_name
                    ),
                    "language": (
                        source.language
                    ),
                    "is_external": (
                        source.is_external
                    ),
                    "url": source.url,
                }
            )


def split_completed_eval(
    expected_dev: int,
    expected_holdout: int,
) -> bool:
    if (
        not EVAL_FULL.is_file()
        or EVAL_FULL.stat().st_size == 0
    ):
        return False

    with EVAL_FULL.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    rows = [
        row
        for row
        in rows
        if any(
            (value or "").strip()
            for value
            in row.values()
        )
    ]

    if not rows:
        return False

    counts = Counter(
        (
            row.get("split")
            or ""
        ).strip()
        for row
        in rows
    )

    if (
        counts.get(
            "dev",
            0,
        )
        != expected_dev
    ):
        raise ValueError(
            f"Expected {expected_dev} "
            "dev questions, found "
            f"{counts.get('dev', 0)}."
        )

    if (
        counts.get(
            "holdout",
            0,
        )
        != expected_holdout
    ):
        raise ValueError(
            f"Expected {expected_holdout} "
            "holdout questions, found "
            f"{counts.get('holdout', 0)}."
        )

    unexpected = (
        set(counts)
        - {
            "dev",
            "holdout",
        }
    )

    if unexpected:
        raise ValueError(
            "Unexpected split values: "
            f"{sorted(unexpected)}"
        )

    for split, output in (
        (
            "dev",
            EVAL_DEV,
        ),
        (
            "holdout",
            EVAL_HOLDOUT,
        ),
    ):
        split_rows = [
            row
            for row
            in rows
            if (
                row.get(
                    "split"
                )
                or ""
            ).strip() == split
        ]

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    EVAL_COLUMNS
                ),
            )

            writer.writeheader()

            for row in split_rows:
                writer.writerow(
                    {
                        column: row.get(
                            column,
                            "",
                        )
                        for column
                        in EVAL_COLUMNS
                    }
                )

    load_eval_questions(
        EVAL_DEV,
        expected_split="dev",
    )

    load_eval_questions(
        EVAL_HOLDOUT,
        expected_split=(
            "holdout"
        ),
    )

    return True


def main(
) -> int:
    args = parse_args()

    EVAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header_only(
        EVAL_FULL,
        overwrite=args.overwrite,
    )

    write_source_reference(
        SOURCE_REFERENCE,
        overwrite=args.overwrite,
    )

    if split_completed_eval(
        args.expected_dev,
        args.expected_holdout,
    ):
        print(
            "Created and validated: "
            f"{EVAL_DEV}"
        )

        print(
            "Created and validated: "
            f"{EVAL_HOLDOUT}"
        )

    else:
        print(
            "Created/confirmed "
            "evaluation template: "
            f"{EVAL_FULL}"
        )

        print(
            "Created/confirmed source "
            "reference: "
            f"{SOURCE_REFERENCE}"
        )

        print(
            "The template is "
            "intentionally empty. Gold "
            "questions and source IDs "
            "must be authored from the "
            "frozen corpus; they are not "
            "generated automatically."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )