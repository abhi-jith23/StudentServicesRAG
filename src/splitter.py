from __future__ import annotations

import hashlib
import json
import re
from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Iterator,
    Sequence,
)

import yaml

from src.config import ChunkingConfig


_FRONT_MATTER_RE = re.compile(
    r"\A---\s*\n"
    r"(?P<yaml>.*?)"
    r"\n---\s*(?:\n|\Z)",
    re.DOTALL,
)

_HEADING_RE = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$"
)

_PAGE_HEADING_RE = re.compile(
    r"^##\s+Page\s+(\d+)\s*$",
    re.IGNORECASE,
)

_LEGAL_CHAPTER_RE = re.compile(
    r"^(Chapitre\s+\d+\.?\s*.+)$",
    re.IGNORECASE,
)

_LEGAL_ARTICLE_RE = re.compile(
    r"^(Art\.\s*\d+"
    r"[A-Za-zÀ-ÿ-]*\.?\s*.+)$",
    re.IGNORECASE,
)

_LEGAL_ANNEX_RE = re.compile(
    r"^(Annexe\s+\d+\s*.+)$",
    re.IGNORECASE,
)

_LIST_LINE_RE = re.compile(
    r"^\s*(?:[-+*]|\d+[.)]|"
    r"[a-zA-Z][.)])\s+\S"
)

_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*"
    r"(?:\|\s*:?-{3,}:?\s*)+"
    r"\|?\s*$"
)

_DOT_LEADER_RE = re.compile(
    r"\.{5,}\s*\d+\s*$"
)

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+"
    r"(?=[A-ZÀ-ÖØ-Ý0-9])"
)

_TOKEN_BUDGET_SAFETY_MARGIN = 8

_NOISE_PATTERNS = (
    re.compile(
        r"^JOURNAL OFFICIEL\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^M[ÉE]MORIAL\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^B\s+\d+\s*-\s*\d+\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^Page\s+\d+\s+"
        r"(?:of|sur)\s+\d+\s*$",
        re.IGNORECASE,
    ),
)


@dataclass(
    frozen=True,
    slots=True,
)
class ParsedDocument:
    metadata: dict[str, Any]
    body: str
    path: Path


@dataclass(
    frozen=True,
    slots=True,
)
class ContentBlock:
    text: str
    kind: str
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None


@dataclass(
    frozen=True,
    slots=True,
)
class ChunkRecord:
    chunk_id: str
    source_id: str
    title: str
    source_url: str
    source_group: str
    audience: str
    degree_level: str
    faculty: str
    programme_name: str
    language: str
    is_external: bool
    document_type: str
    section_path: str
    page_start: int | None
    page_end: int | None
    chunk_index: int
    chunk_text: str
    embedding_text: str
    character_count: int
    token_count: int
    content_sha256: str
    chunking_config: str
    source_path: str

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)


class TokenCounter:
    def __init__(
        self,
        tokenizer: Any,
    ) -> None:
        self.tokenizer = tokenizer

    def encode(
        self,
        text: str,
    ) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )

        return list(
            encoded["input_ids"]
        )

    def count(
        self,
        text: str,
    ) -> int:
        return len(
            self.encode(text)
        )

    def decode(
        self,
        token_ids: Sequence[int],
    ) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    def token_windows(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
    ) -> list[str]:
        token_ids = self.encode(text)

        if len(token_ids) <= max_tokens:
            return [text.strip()]

        step = max(
            1,
            max_tokens
            - max(
                0,
                overlap_tokens,
            ),
        )

        windows: list[str] = []

        for start in range(
            0,
            len(token_ids),
            step,
        ):
            end = min(
                len(token_ids),
                start + max_tokens,
            )

            decoded = self.decode(
                token_ids[start:end]
            )

            if decoded:
                windows.append(decoded)

            if end >= len(token_ids):
                break

        return windows


def parse_cleaned_markdown(
    path: Path | str,
) -> ParsedDocument:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            "Cleaned Markdown file "
            f"not found: {path}"
        )

    raw = path.read_text(
        encoding="utf-8"
    )

    match = _FRONT_MATTER_RE.match(
        raw
    )

    if match is None:
        raise ValueError(
            "Missing YAML front matter "
            f"in {path}"
        )

    metadata = yaml.safe_load(
        match.group("yaml")
    ) or {}

    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            "YAML front matter must "
            f"be a mapping in {path}"
        )

    body = raw[
        match.end():
    ].strip()

    if not body:
        raise ValueError(
            f"Document body is empty in {path}"
        )

    source_id = str(
        metadata.get(
            "source_id",
            "",
        )
    ).strip()

    if not source_id:
        raise ValueError(
            "source_id is missing from "
            f"YAML front matter in {path}"
        )

    return ParsedDocument(
        metadata=metadata,
        body=body,
        path=path,
    )


def _normalise_heading(
    text: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip().strip("#").strip()


def _is_table_start(
    lines: Sequence[str],
    index: int,
) -> bool:
    if index + 1 >= len(lines):
        return False

    first = lines[index].strip()
    second = lines[
        index + 1
    ].strip()

    return (
        first.count("|") >= 2
        and bool(
            _TABLE_SEPARATOR_RE.match(
                second
            )
        )
    )


def _is_noise_line(
    line: str,
) -> bool:
    stripped = line.strip()

    if not stripped:
        return False

    if _DOT_LEADER_RE.search(
        stripped
    ):
        return True

    return any(
        pattern.match(stripped)
        for pattern
        in _NOISE_PATTERNS
    )


def _clean_text_lines(
    lines: Iterable[str],
) -> str:
    kept: list[str] = []

    for line in lines:
        if _is_noise_line(line):
            continue

        kept.append(
            line.rstrip()
        )

    text = "\n".join(
        kept
    ).strip()

    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text


def _legal_heading(
    line: str,
) -> tuple[int, str] | None:
    stripped = line.strip()

    if len(stripped) > 220:
        return None

    if _LEGAL_CHAPTER_RE.match(
        stripped
    ):
        return 2, stripped

    if _LEGAL_ARTICLE_RE.match(
        stripped
    ):
        return 3, stripped

    if _LEGAL_ANNEX_RE.match(
        stripped
    ):
        return 2, stripped

    return None


def parse_markdown_blocks(
    document: ParsedDocument,
) -> list[ContentBlock]:
    lines = document.body.splitlines()

    blocks: list[
        ContentBlock
    ] = []

    headings: dict[
        int,
        str,
    ] = {}

    current_page: int | None = None

    document_title = (
        _normalise_heading(
            str(
                document.metadata.get(
                    "title",
                    "",
                )
            )
        )
    )

    def section_path(
    ) -> tuple[str, ...]:
        return tuple(
            headings[level]
            for level
            in sorted(headings)
        )

    def update_heading(
        level: int,
        text: str,
    ) -> None:
        nonlocal headings

        text = _normalise_heading(
            text
        )

        for existing_level in list(
            headings
        ):
            if existing_level >= level:
                headings.pop(
                    existing_level,
                    None,
                )

        if (
            level == 1
            and document_title
            and (
                text.casefold()
                == document_title.casefold()
            )
        ):
            return

        headings[level] = text

    i = 0

    while i < len(lines):
        raw_line = lines[i]
        stripped = raw_line.strip()

        if not stripped:
            i += 1
            continue

        page_match = (
            _PAGE_HEADING_RE.match(
                stripped
            )
        )

        if page_match:
            current_page = int(
                page_match.group(1)
            )

            i += 1
            continue

        heading_match = (
            _HEADING_RE.match(
                stripped
            )
        )

        if heading_match:
            update_heading(
                len(
                    heading_match.group(1)
                ),
                heading_match.group(2),
            )

            i += 1
            continue

        legal = _legal_heading(
            stripped
        )

        if legal is not None:
            update_heading(
                *legal
            )

            i += 1
            continue

        if _is_table_start(
            lines,
            i,
        ):
            table_lines = [
                lines[i].rstrip(),
                lines[i + 1].rstrip(),
            ]

            i += 2

            while i < len(lines):
                candidate = lines[i]

                if (
                    not candidate.strip()
                    or candidate.count("|") < 2
                ):
                    break

                table_lines.append(
                    candidate.rstrip()
                )

                i += 1

            text = _clean_text_lines(
                table_lines
            )

            if text:
                blocks.append(
                    ContentBlock(
                        text=text,
                        kind="table",
                        section_path=(
                            section_path()
                        ),
                        page_start=(
                            current_page
                        ),
                        page_end=(
                            current_page
                        ),
                    )
                )

            continue

        list_starts_here = bool(
            _LIST_LINE_RE.match(
                raw_line
            )
        ) or (
            stripped
            in {
                "-",
                "*",
                "+",
            }
            and i + 1 < len(lines)
            and lines[
                i + 1
            ].startswith(
                (
                    " ",
                    "\t",
                )
            )
        )

        if list_starts_here:
            if stripped in {
                "-",
                "*",
                "+",
            }:
                list_lines: list[
                    str
                ] = []

            else:
                list_lines = [
                    raw_line.rstrip()
                ]

            i += 1

            while i < len(lines):
                candidate = lines[i]

                if not candidate.strip():
                    break

                if (
                    _HEADING_RE.match(
                        candidate.strip()
                    )
                    or _PAGE_HEADING_RE.match(
                        candidate.strip()
                    )
                ):
                    break

                if _is_table_start(
                    lines,
                    i,
                ):
                    break

                if (
                    _LIST_LINE_RE.match(
                        candidate
                    )
                    or candidate.startswith(
                        (
                            " ",
                            "\t",
                        )
                    )
                ):
                    list_lines.append(
                        candidate.rstrip()
                    )

                    i += 1
                    continue

                break

            text = _clean_text_lines(
                list_lines
            )

            if text:
                blocks.append(
                    ContentBlock(
                        text=text,
                        kind="list",
                        section_path=(
                            section_path()
                        ),
                        page_start=(
                            current_page
                        ),
                        page_end=(
                            current_page
                        ),
                    )
                )

            continue

        paragraph_lines = [
            raw_line.rstrip()
        ]

        i += 1

        while i < len(lines):
            candidate = lines[i]

            candidate_stripped = (
                candidate.strip()
            )

            if not candidate_stripped:
                break

            if _PAGE_HEADING_RE.match(
                candidate_stripped
            ):
                break

            if _HEADING_RE.match(
                candidate_stripped
            ):
                break

            if (
                _legal_heading(
                    candidate_stripped
                )
                is not None
            ):
                break

            if _is_table_start(
                lines,
                i,
            ):
                break

            if _LIST_LINE_RE.match(
                candidate
            ):
                break

            paragraph_lines.append(
                candidate.rstrip()
            )

            i += 1

        text = _clean_text_lines(
            paragraph_lines
        )

        if text:
            blocks.append(
                ContentBlock(
                    text=text,
                    kind="paragraph",
                    section_path=(
                        section_path()
                    ),
                    page_start=(
                        current_page
                    ),
                    page_end=(
                        current_page
                    ),
                )
            )

    return blocks


def _split_table(
    block: ContentBlock,
    counter: TokenCounter,
    max_tokens: int,
) -> list[ContentBlock]:
    lines = block.text.splitlines()

    if (
        len(lines) <= 2
        or counter.count(
            block.text
        ) <= max_tokens
    ):
        return [block]

    header = lines[:2]
    rows = lines[2:]

    output: list[
        ContentBlock
    ] = []

    current = header.copy()

    for row in rows:
        candidate = "\n".join(
            current + [row]
        )

        if (
            len(current) > 2
            and (
                counter.count(
                    candidate
                )
                > max_tokens
            )
        ):
            output.append(
                ContentBlock(
                    text="\n".join(
                        current
                    ),
                    kind="table",
                    section_path=(
                        block.section_path
                    ),
                    page_start=(
                        block.page_start
                    ),
                    page_end=(
                        block.page_end
                    ),
                )
            )

            current = (
                header + [row]
            )

        else:
            current.append(row)

    if len(current) > 2:
        output.append(
            ContentBlock(
                text="\n".join(
                    current
                ),
                kind="table",
                section_path=(
                    block.section_path
                ),
                page_start=(
                    block.page_start
                ),
                page_end=(
                    block.page_end
                ),
            )
        )

    if not output:
        return _split_by_token_windows(
            block,
            counter,
            max_tokens,
        )

    return output


def _split_by_sentences(
    block: ContentBlock,
    counter: TokenCounter,
    max_tokens: int,
) -> list[ContentBlock]:
    if block.kind == "list":
        units = [
            line
            for line
            in block.text.splitlines()
            if line.strip()
        ]

    else:
        units = [
            unit.strip()
            for unit
            in _SENTENCE_SPLIT_RE.split(
                block.text
            )
            if unit.strip()
        ]

    if len(units) <= 1:
        return _split_by_token_windows(
            block,
            counter,
            max_tokens,
        )

    pieces: list[str] = []
    current: list[str] = []

    separator = (
        "\n"
        if block.kind == "list"
        else " "
    )

    for unit in units:
        if (
            counter.count(unit)
            > max_tokens
        ):
            if current:
                pieces.append(
                    separator.join(
                        current
                    ).strip()
                )

                current = []

            pieces.extend(
                counter.token_windows(
                    unit,
                    max_tokens=max_tokens,
                    overlap_tokens=0,
                )
            )

            continue

        candidate = separator.join(
            current + [unit]
        ).strip()

        if (
            current
            and (
                counter.count(
                    candidate
                )
                > max_tokens
            )
        ):
            pieces.append(
                separator.join(
                    current
                ).strip()
            )

            current = [unit]

        else:
            current.append(unit)

    if current:
        pieces.append(
            separator.join(
                current
            ).strip()
        )

    return [
        ContentBlock(
            text=piece,
            kind=block.kind,
            section_path=(
                block.section_path
            ),
            page_start=(
                block.page_start
            ),
            page_end=(
                block.page_end
            ),
        )
        for piece
        in pieces
        if piece
    ]


def _split_by_token_windows(
    block: ContentBlock,
    counter: TokenCounter,
    max_tokens: int,
) -> list[ContentBlock]:
    return [
        ContentBlock(
            text=piece,
            kind=block.kind,
            section_path=(
                block.section_path
            ),
            page_start=(
                block.page_start
            ),
            page_end=(
                block.page_end
            ),
        )
        for piece
        in counter.token_windows(
            block.text,
            max_tokens=max_tokens,
            overlap_tokens=0,
        )
        if piece
    ]


def _expand_oversized_blocks(
    blocks: Sequence[
        ContentBlock
    ],
    counter: TokenCounter,
    max_tokens: int,
) -> list[ContentBlock]:
    expanded: list[
        ContentBlock
    ] = []

    for block in blocks:
        if (
            counter.count(
                block.text
            )
            <= max_tokens
        ):
            expanded.append(block)

        elif block.kind == "table":
            expanded.extend(
                _split_table(
                    block,
                    counter,
                    max_tokens,
                )
            )

        else:
            expanded.extend(
                _split_by_sentences(
                    block,
                    counter,
                    max_tokens,
                )
            )

    return expanded


def _trailing_overlap_blocks(
    blocks: Sequence[
        ContentBlock
    ],
    counter: TokenCounter,
    overlap_tokens: int,
) -> list[ContentBlock]:
    if overlap_tokens <= 0:
        return []

    selected: list[
        ContentBlock
    ] = []

    total = 0

    for block in reversed(
        blocks
    ):
        block_tokens = (
            counter.count(
                block.text
            )
        )

        if (
            selected
            and (
                total + block_tokens
                > overlap_tokens
            )
        ):
            break

        selected.append(block)
        total += block_tokens

        if total >= overlap_tokens:
            break

    selected.reverse()

    return selected


def _combine_pages(
    blocks: Sequence[
        ContentBlock
    ],
) -> tuple[
    int | None,
    int | None,
]:
    pages = [
        page
        for block
        in blocks
        for page
        in (
            block.page_start,
            block.page_end,
        )
        if page is not None
    ]

    if not pages:
        return None, None

    return (
        min(pages),
        max(pages),
    )


def _metadata_prefix(
    metadata: dict[str, Any],
    section_path: str,
    page_start: int | None,
    page_end: int | None,
) -> str:
    fields: list[
        tuple[str, Any]
    ] = [
        (
            "Title",
            metadata.get(
                "title",
                "",
            ),
        ),
        (
            "Source group",
            metadata.get(
                "source_group",
                "",
            ),
        ),
        (
            "Audience",
            metadata.get(
                "audience",
                "",
            ),
        ),
        (
            "Degree level",
            metadata.get(
                "degree_level",
                "",
            ),
        ),
        (
            "Faculty",
            metadata.get(
                "faculty",
                "",
            ),
        ),
        (
            "Programme",
            metadata.get(
                "programme_name",
                "",
            ),
        ),
        (
            "Language",
            metadata.get(
                "language",
                "",
            ),
        ),
        (
            "Section",
            section_path,
        ),
    ]

    if page_start is not None:
        if page_end in {
            None,
            page_start,
        }:
            page_value = str(
                page_start
            )

        else:
            page_value = (
                f"{page_start}-"
                f"{page_end}"
            )

        fields.append(
            (
                "Page",
                page_value,
            )
        )

    lines = [
        f"{label}: {value}"
        for label, value
        in fields
        if str(value).strip()
    ]

    return "\n".join(lines)


def _merge_metadata(
    parsed: ParsedDocument,
    catalog_metadata: dict[
        str,
        Any,
    ],
    manifest_metadata: (
        dict[str, Any]
        | None
    ),
) -> dict[str, Any]:
    merged = dict(
        parsed.metadata
    )

    merged.update(
        {
            key: value
            for key, value
            in catalog_metadata.items()
            if value is not None
        }
    )

    if manifest_metadata:
        for key in (
            "final_url",
            "detected_type",
            "fetch_method",
            "fetched_at",
            "page_count",
            "sha256",
        ):
            value = (
                manifest_metadata.get(
                    key
                )
            )

            if value not in {
                None,
                "",
            }:
                merged[key] = value

    parsed_id = str(
        parsed.metadata.get(
            "source_id",
            "",
        )
    ).strip()

    catalog_id = str(
        catalog_metadata.get(
            "source_id",
            "",
        )
    ).strip()

    if parsed_id != catalog_id:
        raise ValueError(
            "source_id mismatch for "
            f"{parsed.path}: "
            f"front matter={parsed_id!r}, "
            f"catalog={catalog_id!r}"
        )

    return merged


def _group_by_section(
    blocks: Sequence[
        ContentBlock
    ],
) -> Iterator[
    list[ContentBlock]
]:
    current: list[
        ContentBlock
    ] = []

    current_path: (
        tuple[str, ...]
        | None
    ) = None

    for block in blocks:
        if (
            current
            and (
                block.section_path
                != current_path
            )
        ):
            yield current
            current = []

        current.append(block)
        current_path = (
            block.section_path
        )

    if current:
        yield current


def _pack_section_blocks(
    blocks: Sequence[
        ContentBlock
    ],
    *,
    counter: TokenCounter,
    body_budget: int,
    overlap_tokens: int,
    min_tokens: int,
) -> list[
    list[ContentBlock]
]:
    expanded = (
        _expand_oversized_blocks(
            blocks,
            counter,
            body_budget,
        )
    )

    chunks: list[
        list[ContentBlock]
    ] = []

    current: list[
        ContentBlock
    ] = []

    for block in expanded:
        candidate = "\n\n".join(
            item.text
            for item
            in current + [block]
        ).strip()

        if (
            current
            and (
                counter.count(
                    candidate
                )
                > body_budget
            )
        ):
            chunks.append(current)

            current = (
                _trailing_overlap_blocks(
                    current,
                    counter,
                    overlap_tokens,
                )
            )

            while current:
                candidate = (
                    "\n\n".join(
                        item.text
                        for item
                        in current + [block]
                    ).strip()
                )

                if (
                    counter.count(
                        candidate
                    )
                    <= body_budget
                ):
                    break

                current = current[1:]

            current.append(block)

        else:
            current.append(block)

    if current:
        current_tokens = (
            counter.count(
                "\n\n".join(
                    item.text
                    for item
                    in current
                )
            )
        )

        if (
            chunks
            and (
                current_tokens
                < min_tokens
            )
            and (
                counter.count(
                    "\n\n".join(
                        item.text
                        for item
                        in (
                            chunks[-1]
                            + current
                        )
                    )
                )
                <= body_budget
            )
        ):
            chunks[-1] = (
                chunks[-1]
                + current
            )

        else:
            chunks.append(current)

    return chunks


def build_document_chunks(
    path: Path | str,
    *,
    catalog_metadata: dict[
        str,
        Any,
    ],
    manifest_metadata: (
        dict[str, Any]
        | None
    ),
    tokenizer: Any,
    config: ChunkingConfig,
) -> list[ChunkRecord]:
    parsed = parse_cleaned_markdown(
        path
    )

    metadata = _merge_metadata(
        parsed,
        catalog_metadata,
        manifest_metadata,
    )

    blocks = parse_markdown_blocks(
        parsed
    )

    if not blocks:
        raise ValueError(
            "No indexable content "
            f"blocks found in {parsed.path}"
        )

    counter = TokenCounter(
        tokenizer
    )

    records: list[
        ChunkRecord
    ] = []

    source_id = str(
        metadata.get(
            "source_id",
            "",
        )
    ).strip()

    source_path = str(
        parsed.path
    )

    for section_blocks in (
        _group_by_section(
            blocks
        )
    ):
        section_path = (
            " > ".join(
                section_blocks[
                    0
                ].section_path
            ).strip()
        )

        (
            section_page_start,
            section_page_end,
        ) = _combine_pages(
            section_blocks
        )

        prefix = _metadata_prefix(
            metadata,
            section_path,
            section_page_start,
            section_page_end,
        )

        prefix_tokens = (
            counter.count(
                "passage: "
                f"{prefix}\n\n"
            )
        )

        body_budget = (
            config.max_tokens
            - prefix_tokens
            - _TOKEN_BUDGET_SAFETY_MARGIN
        )

        if body_budget < 32:
            raise ValueError(
                "Metadata prefix leaves only "
                f"{body_budget} body tokens "
                f"for {source_id}; reduce "
                "metadata or increase "
                "max_tokens."
            )

        packed = (
            _pack_section_blocks(
                section_blocks,
                counter=counter,
                body_budget=(
                    body_budget
                ),
                overlap_tokens=min(
                    config.overlap_tokens,
                    max(
                        0,
                        body_budget // 3,
                    ),
                ),
                min_tokens=min(
                    config.min_tokens,
                    body_budget,
                ),
            )
        )

        for packed_blocks in packed:
            chunk_text = (
                "\n\n".join(
                    block.text
                    for block
                    in packed_blocks
                ).strip()
            )

            if not chunk_text:
                continue

            (
                page_start,
                page_end,
            ) = _combine_pages(
                packed_blocks
            )

            actual_section_path = (
                " > ".join(
                    packed_blocks[
                        0
                    ].section_path
                ).strip()
            )

            embedding_prefix = (
                _metadata_prefix(
                    metadata,
                    actual_section_path,
                    page_start,
                    page_end,
                )
            )

            embedding_text = (
                f"{embedding_prefix}"
                f"\n\n{chunk_text}"
            ).strip()

            token_count = (
                counter.count(
                    "passage: "
                    f"{embedding_text}"
                )
            )

            if (
                token_count
                > config.max_tokens
            ):
                raise ValueError(
                    "Chunk exceeds "
                    f"{config.max_tokens} "
                    "tokens after prefixing: "
                    f"{source_id}, "
                    "section="
                    f"{actual_section_path!r}, "
                    "tokens="
                    f"{token_count}"
                )

            content_hash = (
                hashlib.sha256(
                    chunk_text.encode(
                        "utf-8"
                    )
                ).hexdigest()
            )

            chunk_index = len(
                records
            )

            chunk_id = (
                f"{source_id}__"
                f"{config.name.upper()}"
                f"__{chunk_index:04d}__"
                f"{content_hash[:8].upper()}"
            )

            document_type = str(
                metadata.get(
                    "detected_type"
                )
                or parsed.metadata.get(
                    "detected_type"
                )
                or parsed.path.parent.name
            ).strip()

            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    source_id=source_id,
                    title=str(
                        metadata.get(
                            "title",
                            "",
                        )
                    ).strip(),
                    source_url=str(
                        metadata.get(
                            "source_url"
                        )
                        or metadata.get(
                            "url"
                        )
                        or metadata.get(
                            "final_url"
                        )
                        or ""
                    ).strip(),
                    source_group=str(
                        metadata.get(
                            "source_group",
                            "",
                        )
                    ).strip(),
                    audience=str(
                        metadata.get(
                            "audience",
                            "",
                        )
                    ).strip(),
                    degree_level=str(
                        metadata.get(
                            "degree_level",
                            "",
                        )
                    ).strip(),
                    faculty=str(
                        metadata.get(
                            "faculty",
                            "",
                        )
                        or ""
                    ).strip(),
                    programme_name=str(
                        metadata.get(
                            "programme_name",
                            "",
                        )
                        or ""
                    ).strip(),
                    language=str(
                        metadata.get(
                            "language",
                            "",
                        )
                    ).strip(),
                    is_external=bool(
                        metadata.get(
                            "is_external",
                            False,
                        )
                    ),
                    document_type=(
                        document_type
                    ),
                    section_path=(
                        actual_section_path
                    ),
                    page_start=(
                        page_start
                    ),
                    page_end=page_end,
                    chunk_index=(
                        chunk_index
                    ),
                    chunk_text=(
                        chunk_text
                    ),
                    embedding_text=(
                        embedding_text
                    ),
                    character_count=len(
                        chunk_text
                    ),
                    token_count=(
                        token_count
                    ),
                    content_sha256=(
                        content_hash
                    ),
                    chunking_config=(
                        config.name
                    ),
                    source_path=(
                        source_path
                    ),
                )
            )

    if not records:
        raise ValueError(
            "No chunks were created "
            f"for {source_id} "
            f"from {parsed.path}"
        )

    return records


def write_chunks_jsonl(
    path: Path | str,
    chunks: Iterable[
        ChunkRecord
        | dict[str, Any]
    ],
) -> int:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for chunk in chunks:
            if isinstance(
                chunk,
                ChunkRecord,
            ):
                payload = (
                    chunk.to_dict()
                )

            else:
                payload = dict(chunk)

            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

    return count


def read_chunks_jsonl(
    path: Path | str,
) -> list[
    dict[str, Any]
]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Chunks file not found: {path}"
        )

    rows: list[
        dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                payload = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON in "
                    f"{path} at line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(
                payload,
                dict,
            ):
                raise ValueError(
                    "Expected a JSON object "
                    f"in {path} at line "
                    f"{line_number}"
                )

            rows.append(payload)

    return rows