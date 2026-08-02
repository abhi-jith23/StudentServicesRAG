from __future__ import annotations

import argparse
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
    CHUNKING_CONFIGS,
    DEFAULT_CANDIDATE_K,
    DEFAULT_EVAL_TOP_K,
    DEFAULT_MAX_CHUNKS_PER_SOURCE,
    EMBEDDING_MODEL_NAME,
    EVAL_DIR,
    index_dir,
    retrieval_error_path,
    retrieval_per_question_path,
    retrieval_summary_path,
)

from src.embedder import (  # noqa: E402
    E5Embedder,
)

from src.eval import (  # noqa: E402
    evaluate_retriever,
    load_eval_questions,
    write_csv,
    write_json,
)

from src.retriever import (  # noqa: E402
    Retriever,
)

from src.vectorstore import (  # noqa: E402
    FaissVectorStore,
)


def parse_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval on "
            "the frozen development "
            "question set."
        )
    )

    parser.add_argument(
        "--config",
        choices=sorted(
            CHUNKING_CONFIGS
        ),
        required=True,
    )

    parser.add_argument(
        "--eval-file",
        type=Path,
        default=(
            EVAL_DIR
            / "eval_dev.csv"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=(
            DEFAULT_EVAL_TOP_K
        ),
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=(
            DEFAULT_CANDIDATE_K
        ),
    )

    parser.add_argument(
        "--max-chunks-per-source",
        type=int,
        default=(
            DEFAULT_MAX_CHUNKS_PER_SOURCE
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    return parser.parse_args()


def write_error_analysis(
    path: Path,
    rows: list[
        dict[str, object]
    ],
) -> None:
    failures = [
        row
        for row
        in rows
        if (
            bool(
                row.get(
                    "answerable"
                )
            )
            and (
                row.get(
                    "hit_at_3"
                )
                == 0
            )
        )
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = (
        list(
            rows[0].keys()
        )
        if rows
        else []
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
        writer.writerows(
            failures
        )


def main(
) -> int:
    args = parse_args()

    questions = (
        load_eval_questions(
            args.eval_file,
            expected_split="dev",
        )
    )

    store = (
        FaissVectorStore.load(
            index_dir(
                args.config
            )
        )
    )

    model_name = str(
        store.manifest.get(
            "embedding_model"
        )
        or EMBEDDING_MODEL_NAME
    )

    embedder = E5Embedder(
        model_name,
        device=args.device,
    )

    retriever = Retriever(
        store=store,
        embedder=embedder,
        candidate_k=(
            args.candidate_k
        ),
        max_chunks_per_source=(
            args
            .max_chunks_per_source
        ),
    )

    rows, summary = (
        evaluate_retriever(
            retriever,
            questions,
            top_k=args.top_k,
        )
    )

    summary.update(
        {
            "chunking_config": (
                args.config
            ),
            "eval_file": str(
                args.eval_file
            ),
            "retrieval_top_k": (
                args.top_k
            ),
            "candidate_k": (
                args.candidate_k
            ),
            (
                "max_chunks_"
                "per_source"
            ): (
                args
                .max_chunks_per_source
            ),
            "embedding_model": (
                model_name
            ),
            "index_manifest": (
                store.manifest
            ),
        }
    )

    per_question_path = (
        retrieval_per_question_path(
            args.config
        )
    )

    summary_path = (
        retrieval_summary_path(
            args.config
        )
    )

    error_path = (
        retrieval_error_path(
            args.config
        )
    )

    write_csv(
        per_question_path,
        rows,
    )

    write_json(
        summary_path,
        summary,
    )

    write_error_analysis(
        error_path,
        rows,
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\nPer-question results: "
        f"{per_question_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        "Failed Hit@3 questions "
        "for manual analysis: "
        f"{error_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )