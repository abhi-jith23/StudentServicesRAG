from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import sys
import time
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

import numpy as np


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
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    chunks_path,
    index_dir,
)

from src.embedder import (  # noqa: E402
    E5Embedder,
)

from src.splitter import (  # noqa: E402
    read_chunks_jsonl,
)

from src.vectorstore import (  # noqa: E402
    FaissVectorStore,
)


def parse_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed a chunk file and "
            "build an exact FAISS "
            "IndexFlatIP index."
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
        "--batch-size",
        type=int,
        default=(
            EMBEDDING_BATCH_SIZE
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def package_version(
    name: str,
) -> str:
    try:
        return (
            importlib.metadata
            .version(name)
        )

    except (
        importlib.metadata
        .PackageNotFoundError
    ):
        return "not-installed"


def main(
) -> int:
    args = parse_args()

    input_path = chunks_path(
        args.config
    )

    output_dir = index_dir(
        args.config
    )

    if (
        output_dir.exists()
        and any(
            output_dir.iterdir()
        )
        and not args.overwrite
    ):
        raise FileExistsError(
            "Index directory is not "
            f"empty: {output_dir}. "
            "Use --overwrite to rebuild."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = read_chunks_jsonl(
        input_path
    )

    if not chunks:
        raise ValueError(
            f"No chunks found in {input_path}"
        )

    embedding_texts = [
        str(
            row["embedding_text"]
        )
        for row
        in chunks
    ]

    embedder = E5Embedder(
        EMBEDDING_MODEL_NAME,
        device=args.device,
        batch_size=args.batch_size,
    )

    info = embedder.info

    print(
        f"Embedding {len(chunks)} "
        f"chunks with {info.model_name} "
        f"on {info.device} "
        f"(dimension={info.dimension})"
    )

    started = (
        time.perf_counter()
    )

    embeddings = (
        embedder.encode_passages(
            embedding_texts,
            show_progress_bar=True,
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - started
    )

    manifest = {
        "created_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "chunking_config": (
            args.config
        ),
        "chunks_file": str(
            input_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "chunks_file_sha256": (
            sha256(
                input_path
            )
        ),
        "chunk_count": len(
            chunks
        ),
        "document_count": len(
            {
                row["source_id"]
                for row
                in chunks
            }
        ),
        "embedding_model": (
            info.model_name
        ),
        "embedding_dimension": (
            info.dimension
        ),
        (
            "model_max_"
            "sequence_length"
        ): info.max_sequence_length,
        "device": info.device,
        "batch_size": (
            args.batch_size
        ),
        (
            "normalised_"
            "embeddings"
        ): True,
        "index_type": (
            "IndexFlatIP"
        ),
        "similarity": (
            "cosine_via_normalised_"
            "inner_product"
        ),
        "embedding_seconds": (
            elapsed_seconds
        ),
        "package_versions": {
            (
                "sentence-transformers"
            ): package_version(
                "sentence-transformers"
            ),
            "transformers": (
                package_version(
                    "transformers"
                )
            ),
            "torch": package_version(
                "torch"
            ),
            "faiss-cpu": (
                package_version(
                    "faiss-cpu"
                )
            ),
            "numpy": package_version(
                "numpy"
            ),
        },
    }

    store = (
        FaissVectorStore.build(
            embeddings,
            chunks,
            manifest=manifest,
        )
    )

    store.save(
        output_dir
    )

    reloaded = (
        FaissVectorStore.load(
            output_dir
        )
    )

    if (
        reloaded.index.ntotal
        != len(chunks)
    ):
        raise ValueError(
            "Reloaded FAISS index "
            "count does not match "
            "chunk count."
        )

    test_query = embeddings[
        0:1
    ]

    before = store.search(
        test_query,
        top_k=min(
            5,
            len(chunks),
        ),
    )

    after = reloaded.search(
        test_query,
        top_k=min(
            5,
            len(chunks),
        ),
    )

    before_positions = [
        hit.position
        for hit
        in before
    ]

    after_positions = [
        hit.position
        for hit
        in after
    ]

    if (
        before_positions
        != after_positions
    ):
        raise ValueError(
            "FAISS save/reload "
            "consistency check failed: "
            f"before={before_positions}, "
            f"after={after_positions}"
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    print(
        f"Index written to: {output_dir}"
    )

    print(
        "index.ntotal: "
        f"{reloaded.index.ntotal}"
    )

    print(
        "Embedding norm range: "
        f"{norms.min():.6f}–"
        f"{norms.max():.6f}"
    )

    print(
        "Embedding time: "
        f"{elapsed_seconds:.2f} seconds"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )