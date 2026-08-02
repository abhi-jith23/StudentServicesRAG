from __future__ import annotations

import argparse
import json
import random
import sys
from collections import (
    Counter,
    defaultdict,
)
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


from src.config import (  # noqa: E402
    CHUNKING_CONFIGS,
    chunk_review_path,
    chunk_stats_path,
    chunks_path,
    get_chunking_config,
)

from src.splitter import (  # noqa: E402
    read_chunks_jsonl,
)


def parse_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate chunk files and "
            "generate deterministic "
            "review samples."
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
        "--samples-per-category",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def select_samples(
    rows: list[
        dict[str, Any]
    ],
    *,
    samples_per_category: int,
    seed: int,
) -> list[
    tuple[
        str,
        dict[str, Any],
    ]
]:
    rng = random.Random(
        seed
    )

    categories: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        categories[
            "source_group:"
            f"{row.get('source_group', '')}"
        ].append(row)

        categories[
            "document_type:"
            f"{row.get('document_type', '')}"
        ].append(row)

        if row.get("faculty"):
            categories[
                "faculty:"
                f"{row['faculty']}"
            ].append(row)

        if row.get(
            "programme_name"
        ):
            categories[
                "programme_specific"
            ].append(row)

        if (
            row.get("language")
            == "fr"
        ):
            categories[
                "french_source"
            ].append(row)

        if (
            row.get(
                "page_start"
            )
            is not None
        ):
            categories[
                "page_aware_pdf"
            ].append(row)

        if "|" in str(
            row.get(
                "chunk_text",
                "",
            )
        ):
            categories[
                "table_like"
            ].append(row)

    priority = [
        "document_type:html",
        "document_type:pdf",
        "programme_specific",
        "page_aware_pdf",
        "french_source",
        "table_like",
    ]

    priority.extend(
        key
        for key
        in sorted(categories)
        if (
            key.startswith(
                "source_group:"
            )
            and key not in priority
        )
    )

    priority.extend(
        key
        for key
        in sorted(categories)
        if key.startswith(
            "faculty:"
        )
    )

    selected: list[
        tuple[
            str,
            dict[str, Any],
        ]
    ] = []

    seen_ids: set[str] = set()

    for category in priority:
        candidates = categories.get(
            category,
            [],
        )

        if not candidates:
            continue

        candidates = (
            candidates.copy()
        )

        rng.shuffle(
            candidates
        )

        count = 0

        for row in candidates:
            chunk_id = str(
                row.get(
                    "chunk_id",
                    "",
                )
            )

            if chunk_id in seen_ids:
                continue

            selected.append(
                (
                    category,
                    row,
                )
            )

            seen_ids.add(
                chunk_id
            )

            count += 1

            if (
                count
                >= samples_per_category
            ):
                break

    return selected


def validate(
    rows: list[
        dict[str, Any]
    ],
    config_name: str,
) -> list[str]:
    config = get_chunking_config(
        config_name
    )

    errors: list[str] = []

    ids = [
        str(
            row.get(
                "chunk_id",
                "",
            )
        )
        for row
        in rows
    ]

    duplicates = [
        item
        for item, count
        in Counter(ids).items()
        if count > 1
    ]

    if duplicates:
        errors.append(
            "Duplicate chunk IDs: "
            f"{duplicates[:10]}"
        )

    for row in rows:
        chunk_id = row.get(
            "chunk_id",
            "<missing>",
        )

        for field in (
            "source_id",
            "title",
            "source_url",
            "chunk_text",
            "embedding_text",
        ):
            if not str(
                row.get(
                    field,
                    "",
                )
            ).strip():
                errors.append(
                    f"{chunk_id}: "
                    f"missing {field}"
                )

        token_count = int(
            row.get(
                "token_count",
                0,
            )
        )

        if (
            token_count
            > config.max_tokens
        ):
            errors.append(
                f"{chunk_id}: "
                "token_count="
                f"{token_count} exceeds "
                f"{config.max_tokens}"
            )

        if token_count <= 0:
            errors.append(
                f"{chunk_id}: invalid "
                "token_count="
                f"{token_count}"
            )

    return errors


def write_review(
    output: Path,
    *,
    rows: list[
        dict[str, Any]
    ],
    stats: dict[
        str,
        Any,
    ],
    samples: list[
        tuple[
            str,
            dict[str, Any],
        ]
    ],
    errors: list[str],
) -> None:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: list[str] = [
        (
            "# Chunk Review — "
            f"{stats.get('chunking_config', '')}"
        ),
        "",
        "## Summary",
        "",
        (
            "- Documents: "
            f"{stats.get('document_count', '')}"
        ),
        (
            "- Chunks: "
            f"{stats.get('chunk_count', len(rows))}"
        ),
        (
            "- Token range: "
            f"{stats.get('minimum_tokens', '')}"
            "–"
            f"{stats.get('maximum_tokens', '')}"
        ),
        (
            "- Median tokens: "
            f"{stats.get('median_tokens', '')}"
        ),
        (
            "- Duplicate content hashes: "
            f"{stats.get('duplicate_content_hash_count', '')}"
        ),
        "",
        "## Automated validation",
        "",
    ]

    if errors:
        lines.extend(
            f"- WARNING: {error}"
            for error
            in errors
        )

    else:
        lines.append(
            "- No fatal automated "
            "validation errors."
        )

    lines.extend(
        [
            "",
            "## Manual review checklist",
            "",
            (
                "For each sample, check "
                "heading/section context, "
                "programme identity, "
                "list/table integrity, page "
                "metadata, navigation noise, "
                "and whether the chunk is "
                "understandable on its own."
            ),
            "",
        ]
    )

    for index, (
        category,
        row,
    ) in enumerate(
        samples,
        start=1,
    ):
        lines.extend(
            [
                (
                    f"### Sample {index}: "
                    f"{row.get('chunk_id', '')}"
                ),
                "",
                (
                    "- Category: "
                    f"`{category}`"
                ),
                (
                    "- Source: "
                    f"`{row.get('source_id', '')}`"
                    " — "
                    f"{row.get('title', '')}"
                ),
                (
                    "- Group: "
                    f"`{row.get('source_group', '')}`"
                ),
                (
                    "- Faculty/programme: "
                    f"`{row.get('faculty', '')}` / "
                    f"`{row.get('programme_name', '')}`"
                ),
                (
                    "- Section: "
                    f"`{row.get('section_path', '')}`"
                ),
                (
                    "- Page: "
                    f"`{row.get('page_start', '')}`"
                    "–"
                    f"`{row.get('page_end', '')}`"
                ),
                (
                    "- Tokens: "
                    f"`{row.get('token_count', '')}`"
                ),
                "",
                "```text",
                str(
                    row.get(
                        "chunk_text",
                        "",
                    )
                ),
                "```",
                "",
                "Review notes:",
                "",
                (
                    "- [ ] Understandable "
                    "without opening the page"
                ),
                (
                    "- [ ] Correct section and "
                    "programme metadata"
                ),
                (
                    "- [ ] No broken list/table"
                ),
                (
                    "- [ ] No navigation/"
                    "header/footer noise"
                ),
                (
                    "- [ ] No missing exception "
                    "or boundary context"
                ),
                "",
            ]
        )

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main(
) -> int:
    args = parse_args()

    rows = read_chunks_jsonl(
        chunks_path(
            args.config
        )
    )

    stats_file = chunk_stats_path(
        args.config
    )

    if not stats_file.is_file():
        raise FileNotFoundError(
            "Chunk statistics file "
            f"not found: {stats_file}"
        )

    stats = json.loads(
        stats_file.read_text(
            encoding="utf-8"
        )
    )

    errors = validate(
        rows,
        args.config,
    )

    samples = select_samples(
        rows,
        samples_per_category=(
            args.samples_per_category
        ),
        seed=args.seed,
    )

    output = chunk_review_path(
        args.config
    )

    write_review(
        output,
        rows=rows,
        stats=stats,
        samples=samples,
        errors=errors,
    )

    print(
        f"Wrote review file: {output}"
    )

    print(
        f"Selected {len(samples)} "
        "chunks for manual inspection."
    )

    if errors:
        print(
            "Automated validation "
            f"produced {len(errors)} "
            "issue(s)."
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )