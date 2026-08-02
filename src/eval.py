from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Sequence,
)

from src.retriever import (
    Retriever,
)


EVAL_COLUMNS = (
    "question_id",
    "question",
    "answerable",
    "audience",
    "student_stage",
    "degree_level",
    "source_group",
    "programme_name",
    "faculty",
    "requires_external",
    "gold_source_ids",
    "gold_source_mode",
    "gold_answer_brief",
    "question_type",
    "split",
    "notes",
)

REQUIRED_EVAL_COLUMNS = (
    "question_id",
    "question",
    "answerable",
    "gold_source_ids",
    "gold_source_mode",
    "split",
)

_TRUE_VALUES = {
    "true",
    "1",
    "yes",
    "y",
}

_FALSE_VALUES = {
    "false",
    "0",
    "no",
    "n",
    "",
}

_GOLD_SOURCE_MODES = {
    "any",
    "all",
}


@dataclass(
    frozen=True,
    slots=True,
)
class EvalQuestion:
    question_id: str
    question: str
    answerable: bool
    audience: str
    student_stage: str
    degree_level: str
    source_group: str
    programme_name: str
    faculty: str
    requires_external: bool
    gold_source_ids: tuple[
        str,
        ...,
    ]
    gold_source_mode: str
    gold_answer_brief: str
    question_type: str
    split: str
    notes: str


def parse_bool(
    value: str,
    *,
    field_name: str,
    row_number: int,
) -> bool:
    normalised = (
        value or ""
    ).strip().casefold()

    if normalised in _TRUE_VALUES:
        return True

    if normalised in _FALSE_VALUES:
        return False

    raise ValueError(
        f"Row {row_number}: "
        f"{field_name} must be "
        "TRUE/FALSE; received "
        f"{value!r}."
    )


def parse_source_ids(
    value: str,
) -> tuple[str, ...]:
    normalised = (
        value or ""
    ).replace(
        ";",
        "|",
    ).replace(
        ",",
        "|",
    )

    values = [
        item.strip()
        for item
        in normalised.split("|")
        if item.strip()
    ]

    return tuple(
        dict.fromkeys(
            values
        )
    )


def load_eval_questions(
    path: Path | str,
    *,
    expected_split: (
        str | None
    ) = None,
) -> list[EvalQuestion]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            "Evaluation file not found: "
            f"{path}. Create it from "
            "the generated template "
            "before running retrieval "
            "evaluation."
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        columns = tuple(
            reader.fieldnames or ()
        )

        missing = [
            column
            for column
            in REQUIRED_EVAL_COLUMNS
            if column not in columns
        ]

        if missing:
            raise ValueError(
                f"{path} is missing "
                "required columns: "
                f"{', '.join(missing)}"
            )

        questions: list[
            EvalQuestion
        ] = []

        seen_ids: set[str] = set()
        errors: list[str] = []

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            cleaned = {
                column: (
                    row.get(column)
                    or ""
                ).strip()
                for column
                in EVAL_COLUMNS
            }

            if not any(
                cleaned.values()
            ):
                continue

            try:
                answerable = (
                    parse_bool(
                        cleaned[
                            "answerable"
                        ],
                        field_name=(
                            "answerable"
                        ),
                        row_number=(
                            row_number
                        ),
                    )
                )

                requires_external = (
                    parse_bool(
                        cleaned[
                            "requires_external"
                        ],
                        field_name=(
                            "requires_external"
                        ),
                        row_number=(
                            row_number
                        ),
                    )
                )

            except ValueError as exc:
                errors.append(
                    str(exc)
                )

                continue

            question_id = cleaned[
                "question_id"
            ]

            question = cleaned[
                "question"
            ]

            split = cleaned[
                "split"
            ]

            gold_source_ids = (
                parse_source_ids(
                    cleaned[
                        "gold_source_ids"
                    ]
                )
            )

            gold_source_mode = (
                cleaned[
                    "gold_source_mode"
                ]
                .strip()
                .casefold()
            )

            if not gold_source_mode:
                if (
                    cleaned[
                        "question_type"
                    ]
                    == "multi_source"
                ):
                    gold_source_mode = "all"

                else:
                    gold_source_mode = "any"

            if (
                gold_source_mode
                not in _GOLD_SOURCE_MODES
            ):
                errors.append(
                    f"Row {row_number}: "
                    "gold_source_mode must be "
                    "'any' or 'all'; received "
                    f"{gold_source_mode!r}."
                )

            if not question_id:
                errors.append(
                    f"Row {row_number}: "
                    "question_id is empty."
                )

            elif question_id in seen_ids:
                errors.append(
                    f"Row {row_number}: "
                    "duplicate question_id "
                    f"{question_id!r}."
                )

            if not question:
                errors.append(
                    f"Row {row_number}: "
                    "question is empty."
                )

            if (
                expected_split
                and split
                != expected_split
            ):
                errors.append(
                    f"Row {row_number}: "
                    "expected split="
                    f"{expected_split!r}, "
                    f"received {split!r}."
                )

            if (
                answerable
                and not gold_source_ids
            ):
                errors.append(
                    f"Row {row_number}: "
                    "answerable question "
                    "has no gold_source_ids."
                )

            if (
                not answerable
                and gold_source_ids
            ):
                errors.append(
                    f"Row {row_number}: "
                    "unanswerable question "
                    "must not list "
                    "gold_source_ids."
                )

            seen_ids.add(
                question_id
            )

            questions.append(
                EvalQuestion(
                    question_id=(
                        question_id
                    ),
                    question=question,
                    answerable=(
                        answerable
                    ),
                    audience=cleaned[
                        "audience"
                    ],
                    student_stage=cleaned[
                        "student_stage"
                    ],
                    degree_level=cleaned[
                        "degree_level"
                    ],
                    source_group=cleaned[
                        "source_group"
                    ],
                    programme_name=cleaned[
                        "programme_name"
                    ],
                    faculty=cleaned[
                        "faculty"
                    ],
                    requires_external=(
                        requires_external
                    ),
                    gold_source_ids=(
                        gold_source_ids
                    ),
                    gold_source_mode=(
                        gold_source_mode
                    ),
                    gold_answer_brief=(
                        cleaned[
                            "gold_answer_brief"
                        ]
                    ),
                    question_type=(
                        cleaned[
                            "question_type"
                        ]
                    ),
                    split=split,
                    notes=cleaned[
                        "notes"
                    ],
                )
            )

    if errors:
        raise ValueError(
            "Evaluation file validation "
            f"failed for {path}:\n"
            + "\n".join(
                f"- {error}"
                for error
                in errors
            )
        )

    if not questions:
        raise ValueError(
            "Evaluation file contains "
            f"no questions: {path}"
        )

    return questions


def reciprocal_rank(
    retrieved_source_ids: (
        Sequence[str]
    ),
    gold_source_ids: set[str],
) -> float:
    for rank, source_id in enumerate(
        retrieved_source_ids,
        start=1,
    ):
        if (
            source_id
            in gold_source_ids
        ):
            return 1.0 / rank

    return 0.0


def hit_at_k(
    retrieved_source_ids: (
        Sequence[str]
    ),
    gold_source_ids: set[str],
    k: int,
) -> int:
    return int(
        any(
            source_id
            in gold_source_ids
            for source_id
            in retrieved_source_ids[:k]
        )
    )


def recall_at_k(
    retrieved_source_ids: Sequence[str],
    gold_source_ids: set[str],
    k: int,
    *,
    mode: str,
) -> float:
    if not gold_source_ids:
        return 0.0

    retrieved = set(
        retrieved_source_ids[:k]
    )

    overlap = (
        retrieved
        & gold_source_ids
    )

    if mode == "any":
        return float(
            bool(overlap)
        )

    if mode == "all":
        return (
            len(overlap)
            / len(gold_source_ids)
        )

    raise ValueError(
        "Unsupported gold source "
        f"mode: {mode!r}"
    )


def _mean(
    values: Iterable[float],
) -> float:
    values = list(values)

    if not values:
        return 0.0

    return float(
        sum(values)
        / len(values)
    )


def evaluate_retriever(
    retriever: Retriever,
    questions: Sequence[
        EvalQuestion
    ],
    *,
    top_k: int = 5,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    rows: list[
        dict[str, Any]
    ] = []

    latencies: list[
        float
    ] = []

    for question in questions:
        started = (
            time.perf_counter()
        )

        route, results = (
            retriever.retrieve(
                question.question,
                top_k=top_k,
            )
        )

        latency_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        latencies.append(
            latency_ms
        )

        retrieved_source_ids = [
            result.source_id
            for result
            in results
        ]

        retrieved_chunk_ids = [
            result.chunk_id
            for result
            in results
        ]

        semantic_scores = [
            result.semantic_score
            for result
            in results
        ]

        adjusted_scores = [
            result.adjusted_score
            for result
            in results
        ]

        retrieved_groups = [
            result.source_group
            for result
            in results
        ]

        retrieved_programmes = [
            result.programme_name
            for result
            in results
        ]

        retrieved_faculties = [
            result.faculty
            for result
            in results
        ]

        retrieved_external = [
            result.is_external
            for result
            in results
        ]

        gold = set(
            question.gold_source_ids
        )

        first_relevant_rank: (
            int | str
        ) = ""

        for rank, source_id in enumerate(
            retrieved_source_ids,
            start=1,
        ):
            if source_id in gold:
                first_relevant_rank = (
                    rank
                )

                break

        if question.answerable:
            hit1: int | str = (
                hit_at_k(
                    retrieved_source_ids,
                    gold,
                    1,
                )
            )

            hit3: int | str = (
                hit_at_k(
                    retrieved_source_ids,
                    gold,
                    3,
                )
            )

            hit5: int | str = (
                hit_at_k(
                    retrieved_source_ids,
                    gold,
                    5,
                )
            )

            recall5: float | str = (
                recall_at_k(
                    retrieved_source_ids,
                    gold,
                    5,
                    mode=(
                        question
                        .gold_source_mode
                    ),
                )
            )

            rr: float | str = (
                reciprocal_rank(
                    retrieved_source_ids[
                        :top_k
                    ],
                    gold,
                )
            )

        else:
            hit1 = ""
            hit3 = ""
            hit5 = ""
            recall5 = ""
            rr = ""

        group_correct: int | str = ""

        if question.source_group:
            group_correct = int(
                question.source_group
                in retrieved_groups[:3]
            )

        programme_correct: (
            int | str
        ) = ""

        if question.programme_name:
            programme_correct = int(
                question.programme_name
                in retrieved_programmes[
                    :3
                ]
            )

        faculty_correct: (
            int | str
        ) = ""

        if question.faculty:
            faculty_correct = int(
                question.faculty
                in retrieved_faculties[
                    :3
                ]
            )

        external_correct: (
            int | str
        ) = ""

        if question.requires_external:
            external_correct = int(
                any(
                    retrieved_external[
                        :3
                    ]
                )
            )

        rows.append(
            {
                "question_id": (
                    question.question_id
                ),
                "question": (
                    question.question
                ),
                "answerable": (
                    question.answerable
                ),
                "question_type": (
                    question.question_type
                ),
                "gold_source_ids": (
                    "|".join(
                        question
                        .gold_source_ids
                    )
                ),
                "gold_source_mode": (
                    question
                    .gold_source_mode
                ),
                "gold_source_group": (
                    question.source_group
                ),
                "gold_programme_name": (
                    question.programme_name
                ),
                "gold_faculty": (
                    question.faculty
                ),
                "requires_external": (
                    question
                    .requires_external
                ),
                "detected_programme": (
                    route.programme_name
                ),
                "detected_faculty": (
                    route.faculty
                ),
                "detected_degree_level": (
                    route.degree_level
                ),
                "detected_audience": (
                    route.audience
                ),
                "detected_source_groups": (
                    "|".join(
                        route
                        .preferred_source_groups
                    )
                ),
                "matched_aliases": (
                    "|".join(
                        route
                        .matched_aliases
                    )
                ),
                "retrieved_source_ids": (
                    "|".join(
                        retrieved_source_ids
                    )
                ),
                "retrieved_chunk_ids": (
                    "|".join(
                        retrieved_chunk_ids
                    )
                ),
                "retrieved_source_groups": (
                    "|".join(
                        retrieved_groups
                    )
                ),
                "retrieved_semantic_scores": (
                    json.dumps(
                        semantic_scores
                    )
                ),
                "retrieved_adjusted_scores": (
                    json.dumps(
                        adjusted_scores
                    )
                ),
                "first_relevant_rank": (
                    first_relevant_rank
                ),
                "hit_at_1": hit1,
                "hit_at_3": hit3,
                "hit_at_5": hit5,
                "recall_at_5": recall5,
                "reciprocal_rank": rr,
                (
                    "source_group_"
                    "accuracy_at_3"
                ): group_correct,
                (
                    "programme_"
                    "accuracy_at_3"
                ): programme_correct,
                (
                    "faculty_"
                    "accuracy_at_3"
                ): faculty_correct,
                (
                    "external_source_"
                    "accuracy_at_3"
                ): external_correct,
                "top_1_semantic_score": (
                    semantic_scores[0]
                    if semantic_scores
                    else ""
                ),
                (
                    "top_3_mean_"
                    "semantic_score"
                ): (
                    _mean(
                        semantic_scores[
                            :3
                        ]
                    )
                    if semantic_scores
                    else ""
                ),
                "top_1_top_2_gap": (
                    (
                        semantic_scores[0]
                        - semantic_scores[1]
                    )
                    if (
                        len(
                            semantic_scores
                        )
                        >= 2
                    )
                    else ""
                ),
                "latency_ms": (
                    latency_ms
                ),
                (
                    "manual_error_"
                    "category"
                ): "",
                "manual_notes": "",
            }
        )

    answerable_rows = [
        row
        for row
        in rows
        if row["answerable"]
    ]

    unanswerable_rows = [
        row
        for row
        in rows
        if not row["answerable"]
    ]

    def numeric(
        column: str,
        subset: Sequence[
            dict[str, Any]
        ],
    ) -> list[float]:
        return [
            float(
                row[column]
            )
            for row
            in subset
            if row.get(
                column
            ) not in {
                None,
                "",
            }
        ]

    summary = {
        "question_count": len(
            rows
        ),
        (
            "answerable_"
            "question_count"
        ): len(
            answerable_rows
        ),
        (
            "unanswerable_"
            "question_count"
        ): len(
            unanswerable_rows
        ),
        "hit_at_1": _mean(
            numeric(
                "hit_at_1",
                answerable_rows,
            )
        ),
        "hit_at_3": _mean(
            numeric(
                "hit_at_3",
                answerable_rows,
            )
        ),
        "hit_at_5": _mean(
            numeric(
                "hit_at_5",
                answerable_rows,
            )
        ),
        "mrr_at_5": _mean(
            numeric(
                "reciprocal_rank",
                answerable_rows,
            )
        ),
        "recall_at_5": _mean(
            numeric(
                "recall_at_5",
                answerable_rows,
            )
        ),
        (
            "source_group_"
            "accuracy_at_3"
        ): _mean(
            numeric(
                (
                    "source_group_"
                    "accuracy_at_3"
                ),
                rows,
            )
        ),
        (
            "programme_"
            "accuracy_at_3"
        ): _mean(
            numeric(
                (
                    "programme_"
                    "accuracy_at_3"
                ),
                rows,
            )
        ),
        (
            "faculty_"
            "accuracy_at_3"
        ): _mean(
            numeric(
                (
                    "faculty_"
                    "accuracy_at_3"
                ),
                rows,
            )
        ),
        (
            "external_source_"
            "accuracy_at_3"
        ): _mean(
            numeric(
                (
                    "external_source_"
                    "accuracy_at_3"
                ),
                rows,
            )
        ),
        "mean_latency_ms": (
            _mean(
                latencies
            )
        ),
        "median_latency_ms": (
            float(
                statistics.median(
                    latencies
                )
            )
            if latencies
            else 0.0
        ),
        (
            "unanswerable_top_1_"
            "score_mean"
        ): _mean(
            numeric(
                (
                    "top_1_"
                    "semantic_score"
                ),
                unanswerable_rows,
            )
        ),
    }

    return rows, summary


def write_csv(
    path: Path | str,
    rows: Sequence[
        dict[str, Any]
    ],
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            "Cannot write an empty "
            "evaluation result CSV."
        )

    fieldnames = list(
        rows[0].keys()
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


def write_json(
    path: Path | str,
    payload: dict[str, Any],
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )