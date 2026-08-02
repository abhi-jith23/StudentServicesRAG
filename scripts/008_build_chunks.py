from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


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
    FETCH_MANIFEST_CSV,
    SOURCES_CSV,
)

from src.config import (  # noqa: E402
    CHUNKING_CONFIGS,
    EMBEDDING_MODEL_NAME,
    chunk_stats_path,
    chunks_path,
    get_chunking_config,
)

from src.embedder import (  # noqa: E402
    E5Embedder,
)

from src.source_catalog import (  # noqa: E402
    load_sources,
)

from src.splitter import (  # noqa: E402
    build_document_chunks,
    write_chunks_jsonl,
)


def parse_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build structure-aware "
            "chunks from the validated "
            "cleaned corpus."
        )
    )

    parser.add_argument(
        "--config",
        choices=[
            *sorted(
                CHUNKING_CONFIGS
            ),
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def read_manifest(
    path: Path,
) -> dict[
    str,
    dict[str, str],
]:
    if not path.is_file():
        raise FileNotFoundError(
            "Fetch manifest not found: "
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = [
            {
                key: (
                    value or ""
                ).strip()
                for key, value
                in row.items()
            }
            for row
            in csv.DictReader(
                handle
            )
        ]

    by_id: dict[
        str,
        dict[str, str],
    ] = {}

    for row in rows:
        source_id = row.get(
            "source_id",
            "",
        )

        if not source_id:
            continue

        if source_id in by_id:
            raise ValueError(
                "Duplicate manifest "
                f"source_id: {source_id}"
            )

        by_id[source_id] = row

    return by_id


def file_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def build_stats(
    chunks: list[
        dict[str, Any]
    ],
    *,
    config_name: str,
    document_count: int,
    source_hashes: dict[
        str,
        str,
    ],
) -> dict[str, Any]:
    token_counts = [
        int(
            row["token_count"]
        )
        for row
        in chunks
    ]

    character_counts = [
        int(
            row[
                "character_count"
            ]
        )
        for row
        in chunks
    ]

    chunks_per_source = Counter(
        str(
            row["source_id"]
        )
        for row
        in chunks
    )

    chunks_per_group = Counter(
        str(
            row["source_group"]
        )
        for row
        in chunks
    )

    chunks_per_faculty = Counter(
        str(
            row.get(
                "faculty",
                "",
            )
            or "<blank>"
        )
        for row
        in chunks
    )

    duplicate_texts = (
        len(chunks)
        - len(
            {
                str(
                    row[
                        "content_sha256"
                    ]
                )
                for row
                in chunks
            }
        )
    )

    duplicate_ids = (
        len(chunks)
        - len(
            {
                str(
                    row["chunk_id"]
                )
                for row
                in chunks
            }
        )
    )

    missing_required = sum(
        1
        for row
        in chunks
        if (
            not row.get(
                "source_id"
            )
            or not row.get(
                "title"
            )
            or not row.get(
                "source_url"
            )
            or not row.get(
                "chunk_text"
            )
        )
    )

    return {
        "chunking_config": (
            config_name
        ),
        (
            "embedding_model_"
            "for_tokenisation"
        ): EMBEDDING_MODEL_NAME,
        "document_count": (
            document_count
        ),
        "chunk_count": len(
            chunks
        ),
        "minimum_tokens": min(
            token_counts
        ),
        "median_tokens": (
            statistics.median(
                token_counts
            )
        ),
        "mean_tokens": (
            statistics.mean(
                token_counts
            )
        ),
        "maximum_tokens": max(
            token_counts
        ),
        "minimum_characters": min(
            character_counts
        ),
        "median_characters": (
            statistics.median(
                character_counts
            )
        ),
        "mean_characters": (
            statistics.mean(
                character_counts
            )
        ),
        "maximum_characters": max(
            character_counts
        ),
        "empty_chunk_count": sum(
            1
            for row
            in chunks
            if not str(
                row[
                    "chunk_text"
                ]
            ).strip()
        ),
        (
            "duplicate_chunk_"
            "id_count"
        ): duplicate_ids,
        (
            "duplicate_content_"
            "hash_count"
        ): duplicate_texts,
        (
            "missing_required_"
            "metadata_count"
        ): missing_required,
        (
            "chunks_with_"
            "page_metadata"
        ): sum(
            1
            for row
            in chunks
            if (
                row.get(
                    "page_start"
                )
                is not None
            )
        ),
        "chunks_per_source": dict(
            sorted(
                chunks_per_source
                .items()
            )
        ),
        (
            "chunks_per_"
            "source_group"
        ): dict(
            sorted(
                chunks_per_group
                .items()
            )
        ),
        "chunks_per_faculty": dict(
            sorted(
                chunks_per_faculty
                .items()
            )
        ),
        (
            "largest_sources_"
            "by_chunk_count"
        ): chunks_per_source.most_common(
            10
        ),
        (
            "smallest_sources_"
            "by_chunk_count"
        ): sorted(
            chunks_per_source.items(),
            key=lambda item: (
                item[1]
            ),
        )[:10],
        (
            "cleaned_source_"
            "sha256"
        ): source_hashes,
    }


def build_one(
    config_name: str,
    *,
    overwrite: bool,
) -> None:
    output_path = chunks_path(
        config_name
    )

    stats_path = chunk_stats_path(
        config_name
    )

    if (
        output_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            "Output already exists: "
            f"{output_path}. Use "
            "--overwrite to rebuild."
        )

    config = get_chunking_config(
        config_name
    )

    tokenizer = (
        E5Embedder.load_tokenizer(
            EMBEDDING_MODEL_NAME
        )
    )

    sources = load_sources(
        SOURCES_CSV,
        approved_only=True,
    )

    manifest_by_id = read_manifest(
        FETCH_MANIFEST_CSV
    )

    all_chunks: list[
        dict[str, Any]
    ] = []

    source_hashes: dict[
        str,
        str,
    ] = {}

    processed_sources = 0

    for source in sources:
        manifest = (
            manifest_by_id.get(
                source.source_id
            )
        )

        if manifest is None:
            raise ValueError(
                "No manifest row for "
                f"{source.source_id}"
            )

        if (
            manifest.get(
                "extraction_status"
            )
            != "extraction_ok"
        ):
            raise ValueError(
                "Source "
                f"{source.source_id} "
                "is not validated for "
                "indexing: status="
                f"{manifest.get('extraction_status')!r}"
            )

        clean_relative = (
            manifest.get(
                "local_clean_path",
                "",
            )
        )

        if not clean_relative:
            raise ValueError(
                "No local_clean_path for "
                f"{source.source_id}"
            )

        clean_path = (
            PROJECT_ROOT
            / clean_relative
        )

        if not clean_path.is_file():
            raise FileNotFoundError(
                "Cleaned file for "
                f"{source.source_id} "
                "does not exist: "
                f"{clean_path}"
            )

        catalog_metadata = (
            source.metadata()
        )

        chunks = (
            build_document_chunks(
                clean_path,
                catalog_metadata=(
                    catalog_metadata
                ),
                manifest_metadata=(
                    manifest
                ),
                tokenizer=tokenizer,
                config=config,
            )
        )

        all_chunks.extend(
            chunk.to_dict()
            for chunk
            in chunks
        )

        source_hashes[
            source.source_id
        ] = file_sha256(
            clean_path
        )

        processed_sources += 1

        print(
            f"[{processed_sources:02d}/"
            f"{len(sources):02d}] "
            f"{source.source_id}: "
            f"{len(chunks)} chunks"
        )

    if not all_chunks:
        raise ValueError(
            "No chunks were produced."
        )

    duplicate_ids = (
        len(all_chunks)
        - len(
            {
                row["chunk_id"]
                for row
                in all_chunks
            }
        )
    )

    if duplicate_ids:
        raise ValueError(
            f"Generated {duplicate_ids} "
            "duplicate chunk IDs."
        )

    write_chunks_jsonl(
        output_path,
        all_chunks,
    )

    stats = build_stats(
        all_chunks,
        config_name=config_name,
        document_count=(
            processed_sources
        ),
        source_hashes=(
            source_hashes
        ),
    )

    stats_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats_path.write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Wrote {len(all_chunks)} "
        f"chunks to {output_path}"
    )

    print(
        "Wrote chunk statistics "
        f"to {stats_path}"
    )


def main(
) -> int:
    args = parse_args()

    if args.config == "all":
        names = sorted(
            CHUNKING_CONFIGS
        )

    else:
        names = [
            args.config
        ]

    for name in names:
        build_one(
            name,
            overwrite=args.overwrite,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )