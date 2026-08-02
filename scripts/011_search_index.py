from __future__ import annotations

import argparse
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
    DEFAULT_MAX_CHUNKS_PER_SOURCE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    index_dir,
)

from src.embedder import (  # noqa: E402
    E5Embedder,
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
            "Search a built FAISS "
            "index and print metadata-"
            "rich results."
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
        "--query",
        default="",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
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


def print_results(
    retriever: Retriever,
    query: str,
    top_k: int,
) -> None:
    route, results = (
        retriever.retrieve(
            query,
            top_k=top_k,
        )
    )

    print("\nQUERY")
    print(query)

    print("\nROUTE")

    for key, value in (
        route.to_dict().items()
    ):
        print(
            f"- {key}: {value}"
        )

    print("\nRESULTS")

    if not results:
        print("No results.")
        return

    for result in results:
        page = ""

        if (
            result.page_start
            is not None
        ):
            if result.page_end in {
                None,
                result.page_start,
            }:
                page = str(
                    result.page_start
                )

            else:
                page = (
                    f"{result.page_start}-"
                    f"{result.page_end}"
                )

        print("=" * 88)

        print(
            f"#{result.rank} | "
            "adjusted="
            f"{result.adjusted_score:.4f}"
            " | semantic="
            f"{result.semantic_score:.4f}"
        )

        print(
            f"Source: {result.source_id}"
            f" — {result.title}"
        )

        print(
            f"URL: {result.source_url}"
        )

        print(
            f"Group: {result.source_group}"
            " | Faculty: "
            f"{result.faculty or '-'}"
            " | Programme: "
            f"{result.programme_name or '-'}"
            " | Page: "
            f"{page or '-'}"
        )

        print(
            "Section: "
            f"{result.section_path or '-'}"
        )

        print(
            "Matched preferences: "
            + (
                ", ".join(
                    result
                    .matched_preferences
                )
                or "-"
            )
        )

        print("-" * 88)
        print(result.chunk_text)
        print()


def main(
) -> int:
    args = parse_args()

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

    if args.query.strip():
        print_results(
            retriever,
            args.query.strip(),
            args.top_k,
        )

        return 0

    print(
        "Interactive retrieval. "
        "Enter a blank line or "
        "Ctrl-D to stop."
    )

    while True:
        try:
            query = input(
                "\nQuestion> "
            ).strip()

        except EOFError:
            print()
            break

        if not query:
            break

        print_results(
            retriever,
            query,
            args.top_k,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )