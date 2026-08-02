from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.collection_config import (
    CLEAN_HTML_DIR,
    CLEAN_PDF_DIR,
    FETCH_MANIFEST_CSV,
    PROJECT_ROOT,
    SOURCES_CSV,
)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"
RESULTS_DIR = PROJECT_ROOT / "results"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_BATCH_SIZE = 32

DEFAULT_TOP_K = 4
DEFAULT_EVAL_TOP_K = 5
DEFAULT_CANDIDATE_K = 40
DEFAULT_MAX_CHUNKS_PER_SOURCE = 1


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    name: str
    max_tokens: int
    overlap_tokens: int
    min_tokens: int


CHUNKING_CONFIGS: dict[str, ChunkingConfig] = {
    "compact": ChunkingConfig(
        name="compact",
        max_tokens=220,
        overlap_tokens=40,
        min_tokens=35,
    ),
    "broad": ChunkingConfig(
        name="broad",
        max_tokens=350,
        overlap_tokens=60,
        min_tokens=50,
    ),
}


def get_chunking_config(name: str) -> ChunkingConfig:
    try:
        return CHUNKING_CONFIGS[name]

    except KeyError as exc:
        valid = ", ".join(
            sorted(CHUNKING_CONFIGS)
        )

        raise ValueError(
            f"Unknown chunking configuration "
            f"{name!r}. Valid values: {valid}."
        ) from exc


def chunks_path(
    config_name: str,
) -> Path:
    return (
        PROCESSED_DIR
        / f"chunks_{config_name}.jsonl"
    )


def chunk_stats_path(
    config_name: str,
) -> Path:
    return (
        PROCESSED_DIR
        / f"chunking_stats_{config_name}.json"
    )


def chunk_review_path(
    config_name: str,
) -> Path:
    return (
        PROCESSED_DIR
        / f"chunk_review_{config_name}.md"
    )


def index_dir(
    config_name: str,
) -> Path:
    return (
        FAISS_INDEX_DIR
        / config_name
    )


def retrieval_per_question_path(
    config_name: str,
) -> Path:
    return (
        RESULTS_DIR
        / (
            f"retrieval_{config_name}"
            "_per_question.csv"
        )
    )


def retrieval_summary_path(
    config_name: str,
) -> Path:
    return (
        RESULTS_DIR
        / (
            f"retrieval_{config_name}"
            "_summary.json"
        )
    )


def retrieval_error_path(
    config_name: str,
) -> Path:
    return (
        RESULTS_DIR
        / (
            f"retrieval_{config_name}"
            "_error_analysis.csv"
        )
    )