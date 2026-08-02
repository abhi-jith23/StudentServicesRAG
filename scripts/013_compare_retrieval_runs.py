from __future__ import annotations

import csv
import json
import sys
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


from src.config import (  # noqa: E402
    RESULTS_DIR,
    retrieval_per_question_path,
    retrieval_summary_path,
)


OUTPUT_SUMMARY = (
    RESULTS_DIR
    / "retrieval_comparison_summary.json"
)

OUTPUT_PER_QUESTION = (
    RESULTS_DIR
    / "retrieval_comparison_per_question.csv"
)

METRICS = (
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "mrr_at_5",
    "recall_at_5",
    "source_group_accuracy_at_3",
    "programme_accuracy_at_3",
    "faculty_accuracy_at_3",
    "external_source_accuracy_at_3",
    "mean_latency_ms",
    "median_latency_ms",
)


def read_json(
    path: Path,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            "Missing result file: "
            f"{path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_csv_by_id(
    path: Path,
) -> dict[
    str,
    dict[str, str],
]:
    if not path.is_file():
        raise FileNotFoundError(
            "Missing result file: "
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    return {
        row["question_id"]: row
        for row
        in rows
    }


def as_float(
    value: (
        str
        | float
        | int
        | None
    ),
) -> float | None:
    if value in {
        None,
        "",
    }:
        return None

    return float(value)


def main(
) -> int:
    compact_summary = (
        read_json(
            retrieval_summary_path(
                "compact"
            )
        )
    )

    broad_summary = (
        read_json(
            retrieval_summary_path(
                "broad"
            )
        )
    )

    metric_comparison: dict[
        str,
        dict[
            str,
            float | None,
        ],
    ] = {}

    for metric in METRICS:
        compact = as_float(
            compact_summary.get(
                metric
            )
        )

        broad = as_float(
            broad_summary.get(
                metric
            )
        )

        metric_comparison[
            metric
        ] = {
            "compact": compact,
            "broad": broad,
            (
                "broad_minus_"
                "compact"
            ): (
                broad - compact
                if (
                    compact is not None
                    and broad is not None
                )
                else None
            ),
        }

    compact_rows = (
        read_csv_by_id(
            retrieval_per_question_path(
                "compact"
            )
        )
    )

    broad_rows = (
        read_csv_by_id(
            retrieval_per_question_path(
                "broad"
            )
        )
    )

    if (
        set(compact_rows)
        != set(broad_rows)
    ):
        raise ValueError(
            "Compact and broad "
            "evaluations contain "
            "different question IDs."
        )

    comparison_rows: list[
        dict[str, object]
    ] = []

    compact_wins = 0
    broad_wins = 0
    ties = 0

    for question_id in sorted(
        compact_rows
    ):
        compact = compact_rows[
            question_id
        ]

        broad = broad_rows[
            question_id
        ]

        compact_rr = (
            as_float(
                compact.get(
                    "reciprocal_rank"
                )
            )
            or 0.0
        )

        broad_rr = (
            as_float(
                broad.get(
                    "reciprocal_rank"
                )
            )
            or 0.0
        )

        if compact_rr > broad_rr:
            winner = "compact"
            compact_wins += 1

        elif broad_rr > compact_rr:
            winner = "broad"
            broad_wins += 1

        else:
            winner = "tie"
            ties += 1

        comparison_rows.append(
            {
                "question_id": (
                    question_id
                ),
                "question": (
                    compact.get(
                        "question",
                        "",
                    )
                ),
                "question_type": (
                    compact.get(
                        "question_type",
                        "",
                    )
                ),
                "gold_source_ids": (
                    compact.get(
                        "gold_source_ids",
                        "",
                    )
                ),
                (
                    "compact_first_"
                    "relevant_rank"
                ): compact.get(
                    "first_relevant_rank",
                    "",
                ),
                (
                    "broad_first_"
                    "relevant_rank"
                ): broad.get(
                    "first_relevant_rank",
                    "",
                ),
                (
                    "compact_hit_at_3"
                ): compact.get(
                    "hit_at_3",
                    "",
                ),
                (
                    "broad_hit_at_3"
                ): broad.get(
                    "hit_at_3",
                    "",
                ),
                (
                    "compact_"
                    "reciprocal_rank"
                ): compact.get(
                    "reciprocal_rank",
                    "",
                ),
                (
                    "broad_"
                    "reciprocal_rank"
                ): broad.get(
                    "reciprocal_rank",
                    "",
                ),
                "compact_sources": (
                    compact.get(
                        "retrieved_source_ids",
                        "",
                    )
                ),
                "broad_sources": (
                    broad.get(
                        "retrieved_source_ids",
                        "",
                    )
                ),
                (
                    "winner_by_"
                    "reciprocal_rank"
                ): winner,
                "manual_preference": "",
                "manual_notes": "",
            }
        )

    summary = {
        "metric_comparison": (
            metric_comparison
        ),
        (
            "question_level_wins_"
            "by_reciprocal_rank"
        ): {
            "compact": (
                compact_wins
            ),
            "broad": broad_wins,
            "ties": ties,
        },
        "selection_note": (
            "Do not select a "
            "configuration from this "
            "file alone. Review failed "
            "questions, programme-"
            "specific questions, "
            "multi-source recall, and "
            "the manual chunk-review "
            "files before freezing "
            "the winner."
        ),
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_SUMMARY.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with OUTPUT_PER_QUESTION.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                comparison_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            comparison_rows
        )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\nSummary comparison: "
        f"{OUTPUT_SUMMARY}"
    )

    print(
        "Per-question comparison: "
        f"{OUTPUT_PER_QUESTION}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )