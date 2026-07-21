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
    FETCH_MANIFEST_CSV,
    HTML_MIN_TEXT_LENGTH,
    PDF_MIN_TEXT_LENGTH,
    REVIEW_DIR,
    SOURCES_CSV,
)
from src.source_catalog import (  # noqa: E402
    load_sources,
)

REPORT_FIELDS = (
    "source_id",
    "title",
    "detected_type",
    "extraction_status",
    "raw_exists",
    "clean_exists",
    "text_length",
    "severity",
    "message",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the completed "
            "source-collection snapshot."
        )
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=SOURCES_CSV,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=FETCH_MANIFEST_CSV,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=(
            REVIEW_DIR
            / "collection_validation.csv"
        ),
    )

    return parser.parse_args()


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


def write_report(
    path: Path,
    rows: list[dict[str, object]],
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
            fieldnames=REPORT_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in REPORT_FIELDS
                }
            )


def to_int(
    value: str,
) -> int:
    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return 0


def main() -> int:
    args = parse_args()

    sources = load_sources(
        args.catalog,
        approved_only=True,
    )

    manifest_rows = read_rows(
        args.manifest
    )

    manifest_by_id = {
        row.get("source_id", ""): row
        for row in manifest_rows
        if row.get("source_id")
    }

    report: list[
        dict[str, object]
    ] = []

    for source in sources:
        entry = manifest_by_id.get(
            source.source_id
        )

        if entry is None:
            report.append(
                {
                    "source_id": (
                        source.source_id
                    ),
                    "title": source.title,
                    "severity": "fatal",
                    "message": (
                        "No manifest entry exists."
                    ),
                }
            )
            continue

        raw_path_value = entry.get(
            "local_raw_path",
            "",
        )

        clean_path_value = entry.get(
            "local_clean_path",
            "",
        )

        raw_path = (
            PROJECT_ROOT / raw_path_value
            if raw_path_value
            else None
        )

        clean_path = (
            PROJECT_ROOT / clean_path_value
            if clean_path_value
            else None
        )

        raw_exists = bool(
            raw_path
            and raw_path.is_file()
        )

        clean_exists = bool(
            clean_path
            and clean_path.is_file()
        )

        detected_type = entry.get(
            "detected_type",
            "",
        )

        status = entry.get(
            "extraction_status",
            "",
        )

        text_length = to_int(
            entry.get(
                "text_length",
                "",
            )
        )

        severity = "ok"
        messages: list[str] = []

        if status in {
            "fetch_error",
            "unsupported_type",
            "blocked_page",
            "extraction_error",
            "",
        }:
            severity = "fatal"

            messages.append(
                "Invalid extraction status: "
                f"{status or '[blank]'}."
            )

        elif status in {
            "needs_manual_review",
            "scanned_pdf_suspected",
        }:
            severity = "warning"

            messages.append(
                "Manual review required: "
                f"{status}."
            )

        if detected_type not in {
            "html",
            "pdf",
        }:
            severity = "fatal"

            messages.append(
                "Unexpected detected type: "
                f"{detected_type or '[blank]'}."
            )

        if not raw_exists:
            severity = "fatal"

            messages.append(
                "Raw cached file is missing."
            )

        if not clean_exists:
            severity = "fatal"

            messages.append(
                "Clean Markdown file is missing."
            )

        minimum = (
            PDF_MIN_TEXT_LENGTH
            if detected_type == "pdf"
            else HTML_MIN_TEXT_LENGTH
        )

        if text_length < minimum:
            if severity != "fatal":
                severity = "warning"

            messages.append(
                "Extracted text is short: "
                f"{text_length} characters "
                f"(review threshold {minimum})."
            )

        if (
            clean_exists
            and clean_path is not None
        ):
            clean_text = clean_path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            if (
                f"source_id: "
                f"{source.source_id}"
                not in clean_text
            ):
                severity = "fatal"

                messages.append(
                    "Clean Markdown front matter "
                    "has the wrong source_id."
                )

            lowered = clean_text.lower()

            if (
                "verify that you're not a robot"
                in lowered
                or "checking your browser"
                in lowered
                or "access denied"
                in lowered
            ):
                severity = "fatal"

                messages.append(
                    "Clean Markdown appears to "
                    "contain a block page."
                )

        if not messages:
            messages.append(
                "Collection output passed "
                "automated validation."
            )

        report.append(
            {
                "source_id": (
                    source.source_id
                ),
                "title": source.title,
                "detected_type": (
                    detected_type
                ),
                "extraction_status": status,
                "raw_exists": raw_exists,
                "clean_exists": clean_exists,
                "text_length": text_length,
                "severity": severity,
                "message": " ".join(
                    messages
                ),
            }
        )

    write_report(
        args.report,
        report,
    )

    counts = Counter(
        str(row.get("severity", ""))
        for row in report
    )

    print(
        "Approved catalogue sources: "
        f"{len(sources)}"
    )

    print(
        f"Manifest rows: "
        f"{len(manifest_rows)}"
    )

    print(f"OK: {counts['ok']}")
    print(f"Warnings: {counts['warning']}")
    print(f"Fatal: {counts['fatal']}")
    print(f"Report: {args.report}")

    if counts["warning"]:
        print("\nWarning sources:")

        for row in report:
            if (
                row.get("severity")
                == "warning"
            ):
                print(
                    f"  {row['source_id']}: "
                    f"{row['message']}"
                )

    if counts["fatal"]:
        print("\nFatal sources:")

        for row in report:
            if (
                row.get("severity")
                == "fatal"
            ):
                print(
                    f"  {row['source_id']}: "
                    f"{row['message']}"
                )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())