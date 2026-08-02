from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Iterable,
    Sequence,
)

import numpy as np

from sentence_transformers import (
    SentenceTransformer,
)

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
)


@dataclass(
    frozen=True,
    slots=True,
)
class EmbeddingInfo:
    model_name: str
    dimension: int
    max_sequence_length: int
    device: str


class E5Embedder:
    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(
        self,
        model_name: str = (
            EMBEDDING_MODEL_NAME
        ),
        *,
        device: str | None = None,
        batch_size: int = (
            EMBEDDING_BATCH_SIZE
        ),
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        self.model = (
            SentenceTransformer(
                model_name,
                device=device,
            )
        )

        self.tokenizer: (
            PreTrainedTokenizerBase
        ) = self.model.tokenizer

    @staticmethod
    def load_tokenizer(
        model_name: str = (
            EMBEDDING_MODEL_NAME
        ),
    ) -> PreTrainedTokenizerBase:
        return (
            AutoTokenizer
            .from_pretrained(
                model_name,
                use_fast=True,
            )
        )

    @property
    def info(
        self,
    ) -> EmbeddingInfo:
        dimension = int(
            self.model
            .get_sentence_embedding_dimension()
        )

        max_length = int(
            self.model.max_seq_length
        )

        device = str(
            self.model.device
        )

        return EmbeddingInfo(
            model_name=(
                self.model_name
            ),
            dimension=dimension,
            max_sequence_length=(
                max_length
            ),
            device=device,
        )

    @classmethod
    def format_query(
        cls,
        text: str,
    ) -> str:
        cleaned = text.strip()

        if not cleaned:
            raise ValueError(
                "Query text cannot be empty."
            )

        if (
            cleaned.casefold()
            .startswith(
                cls.QUERY_PREFIX
            )
        ):
            return cleaned

        return (
            f"{cls.QUERY_PREFIX}"
            f"{cleaned}"
        )

    @classmethod
    def format_passage(
        cls,
        text: str,
    ) -> str:
        cleaned = text.strip()

        if not cleaned:
            raise ValueError(
                "Passage text cannot be empty."
            )

        if (
            cleaned.casefold()
            .startswith(
                cls.PASSAGE_PREFIX
            )
        ):
            return cleaned

        return (
            f"{cls.PASSAGE_PREFIX}"
            f"{cleaned}"
        )

    def token_count(
        self,
        text: str,
        *,
        kind: str,
    ) -> int:
        if kind == "query":
            formatted = (
                self.format_query(
                    text
                )
            )

        elif kind == "passage":
            formatted = (
                self.format_passage(
                    text
                )
            )

        else:
            raise ValueError(
                "kind must be "
                "'query' or 'passage'."
            )

        return len(
            self.tokenizer.encode(
                formatted,
                add_special_tokens=True,
                truncation=False,
            )
        )

    def _validate_lengths(
        self,
        texts: Sequence[str],
    ) -> None:
        max_length = int(
            self.model.max_seq_length
        )

        oversized: list[
            tuple[int, int]
        ] = []

        for index, text in enumerate(
            texts
        ):
            token_count = len(
                self.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
            )

            if token_count > max_length:
                oversized.append(
                    (
                        index,
                        token_count,
                    )
                )

        if oversized:
            preview = ", ".join(
                f"row {index}: "
                f"{count} tokens"
                for index, count
                in oversized[:10]
            )

            raise ValueError(
                f"{len(oversized)} texts "
                "exceed "
                "model.max_seq_length="
                f"{max_length}. "
                f"Examples: {preview}"
            )

    def _encode(
        self,
        texts: Sequence[str],
        *,
        show_progress_bar: bool,
    ) -> np.ndarray:
        if not texts:
            return np.empty(
                (
                    0,
                    int(
                        self.model
                        .get_sentence_embedding_dimension()
                    ),
                ),
                dtype=np.float32,
            )

        self._validate_lengths(
            texts
        )

        embeddings = (
            self.model.encode(
                list(texts),
                batch_size=(
                    self.batch_size
                ),
                show_progress_bar=(
                    show_progress_bar
                ),
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Expected a 2D embedding "
                "matrix, received shape "
                f"{embeddings.shape}."
            )

        if not np.isfinite(
            embeddings
        ).all():
            raise ValueError(
                "Embedding matrix contains "
                "NaN or infinite values."
            )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
        )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-4,
        ):
            raise ValueError(
                "Embeddings were expected "
                "to be L2-normalised, but "
                "norms range from "
                f"{norms.min():.6f} to "
                f"{norms.max():.6f}."
            )

        return np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

    def encode_queries(
        self,
        queries: Iterable[str],
        *,
        show_progress_bar: (
            bool
        ) = False,
    ) -> np.ndarray:
        formatted = [
            self.format_query(
                query
            )
            for query
            in queries
        ]

        return self._encode(
            formatted,
            show_progress_bar=(
                show_progress_bar
            ),
        )

    def encode_passages(
        self,
        passages: Iterable[str],
        *,
        show_progress_bar: (
            bool
        ) = True,
    ) -> np.ndarray:
        formatted = [
            self.format_passage(
                passage
            )
            for passage
            in passages
        ]

        return self._encode(
            formatted,
            show_progress_bar=(
                show_progress_bar
            ),
        )