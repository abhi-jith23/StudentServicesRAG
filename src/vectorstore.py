from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Sequence,
)

import faiss
import numpy as np


@dataclass(
    frozen=True,
    slots=True,
)
class SearchHit:
    position: int
    score: float
    metadata: dict[str, Any]

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "position": self.position,
            "score": self.score,
            **self.metadata,
        }


class FaissVectorStore:
    INDEX_FILENAME = "index.faiss"

    EMBEDDINGS_FILENAME = (
        "embeddings.npy"
    )

    METADATA_FILENAME = (
        "metadata.jsonl"
    )

    MANIFEST_FILENAME = (
        "build_manifest.json"
    )

    def __init__(
        self,
        *,
        index: faiss.Index,
        embeddings: np.ndarray,
        metadata: list[
            dict[str, Any]
        ],
        manifest: (
            dict[str, Any]
            | None
        ) = None,
    ) -> None:
        self.index = index

        self.embeddings = (
            np.ascontiguousarray(
                embeddings,
                dtype=np.float32,
            )
        )

        self.metadata = metadata
        self.manifest = (
            manifest or {}
        )

        self._validate_integrity()

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        metadata: Sequence[
            dict[str, Any]
        ],
        *,
        manifest: (
            dict[str, Any]
            | None
        ) = None,
    ) -> "FaissVectorStore":
        matrix = (
            np.ascontiguousarray(
                embeddings,
                dtype=np.float32,
            )
        )

        if matrix.ndim != 2:
            raise ValueError(
                "Embeddings must be 2D, "
                f"received {matrix.shape}."
            )

        if matrix.shape[0] == 0:
            raise ValueError(
                "Cannot build a FAISS "
                "index from zero embeddings."
            )

        if len(metadata) != matrix.shape[0]:
            raise ValueError(
                "Metadata rows "
                f"({len(metadata)}) do not "
                "match embeddings "
                f"({matrix.shape[0]})."
            )

        if not np.isfinite(
            matrix
        ).all():
            raise ValueError(
                "Embeddings contain NaN "
                "or infinite values."
            )

        norms = np.linalg.norm(
            matrix,
            axis=1,
        )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-4,
        ):
            faiss.normalize_L2(
                matrix
            )

        index = faiss.IndexFlatIP(
            matrix.shape[1]
        )

        index.add(matrix)

        return cls(
            index=index,
            embeddings=matrix,
            metadata=[
                dict(row)
                for row
                in metadata
            ],
            manifest=manifest,
        )

    def _validate_integrity(
        self,
    ) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(
                "Stored embeddings must "
                "be 2D, received "
                f"{self.embeddings.shape}."
            )

        if (
            self.index.d
            != self.embeddings.shape[1]
        ):
            raise ValueError(
                "FAISS dimension "
                f"({self.index.d}) does not "
                "match embeddings "
                f"({self.embeddings.shape[1]})."
            )

        if (
            self.index.ntotal
            != self.embeddings.shape[0]
        ):
            raise ValueError(
                "FAISS index rows "
                f"({self.index.ntotal}) do "
                "not match embeddings "
                f"({self.embeddings.shape[0]})."
            )

        if (
            len(self.metadata)
            != self.embeddings.shape[0]
        ):
            raise ValueError(
                "Metadata rows "
                f"({len(self.metadata)}) do "
                "not match embeddings "
                f"({self.embeddings.shape[0]})."
            )

        if not np.isfinite(
            self.embeddings
        ).all():
            raise ValueError(
                "Stored embeddings contain "
                "NaN or infinite values."
            )

    def save(
        self,
        output_dir: Path | str,
    ) -> None:
        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(
                output_dir
                / self.INDEX_FILENAME
            ),
        )

        np.save(
            output_dir
            / self.EMBEDDINGS_FILENAME,
            self.embeddings,
        )

        with (
            output_dir
            / self.METADATA_FILENAME
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            for row in self.metadata:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        with (
            output_dir
            / self.MANIFEST_FILENAME
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                self.manifest,
                handle,
                ensure_ascii=False,
                indent=2,
            )

            handle.write("\n")

    @classmethod
    def load(
        cls,
        output_dir: Path | str,
    ) -> "FaissVectorStore":
        output_dir = Path(
            output_dir
        )

        index_path = (
            output_dir
            / cls.INDEX_FILENAME
        )

        embeddings_path = (
            output_dir
            / cls.EMBEDDINGS_FILENAME
        )

        metadata_path = (
            output_dir
            / cls.METADATA_FILENAME
        )

        manifest_path = (
            output_dir
            / cls.MANIFEST_FILENAME
        )

        for path in (
            index_path,
            embeddings_path,
            metadata_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    "Required index file "
                    f"not found: {path}"
                )

        index = faiss.read_index(
            str(index_path)
        )

        embeddings = np.load(
            embeddings_path
        ).astype(
            np.float32,
            copy=False,
        )

        metadata: list[
            dict[str, Any]
        ] = []

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                if not line.strip():
                    continue

                payload = json.loads(
                    line
                )

                if not isinstance(
                    payload,
                    dict,
                ):
                    raise ValueError(
                        "Expected JSON object "
                        f"in {metadata_path} "
                        "line "
                        f"{line_number}."
                    )

                metadata.append(
                    payload
                )

        manifest: dict[
            str,
            Any,
        ] = {}

        if manifest_path.is_file():
            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )

        return cls(
            index=index,
            embeddings=embeddings,
            metadata=metadata,
            manifest=manifest,
        )

    def positions_matching(
        self,
        predicate: Callable[
            [dict[str, Any]],
            bool,
        ],
    ) -> np.ndarray:
        positions = [
            index
            for index, row
            in enumerate(
                self.metadata
            )
            if predicate(row)
        ]

        return np.asarray(
            positions,
            dtype=np.int64,
        )

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        allowed_positions: (
            Sequence[int]
            | np.ndarray
            | None
        ) = None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []

        query = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if query.ndim == 1:
            query = query.reshape(
                1,
                -1,
            )

        expected_shape = (
            1,
            self.embeddings.shape[1],
        )

        if query.shape != expected_shape:
            raise ValueError(
                "Query vector must have "
                f"shape {expected_shape}, "
                f"received {query.shape}."
            )

        if not np.isfinite(
            query
        ).all():
            raise ValueError(
                "Query vector contains NaN "
                "or infinite values."
            )

        query = (
            np.ascontiguousarray(
                query,
                dtype=np.float32,
            )
        )

        norm = np.linalg.norm(
            query,
            axis=1,
        )

        if not np.allclose(
            norm,
            1.0,
            atol=1e-4,
        ):
            faiss.normalize_L2(
                query
            )

        if allowed_positions is None:
            k = min(
                top_k,
                int(
                    self.index.ntotal
                ),
            )

            scores, positions = (
                self.index.search(
                    query,
                    k,
                )
            )

            ordered = [
                (
                    int(position),
                    float(score),
                )
                for position, score
                in zip(
                    positions[0],
                    scores[0],
                )
                if position >= 0
            ]

        else:
            allowed = np.asarray(
                allowed_positions,
                dtype=np.int64,
            )

            allowed = allowed[
                (allowed >= 0)
                & (
                    allowed
                    < len(self.metadata)
                )
            ]

            if allowed.size == 0:
                return []

            subset_scores = (
                self.embeddings[
                    allowed
                ]
                @ query[0]
            )

            k = min(
                top_k,
                allowed.size,
            )

            local_order = np.argsort(
                -subset_scores,
                kind="stable",
            )[:k]

            ordered = [
                (
                    int(
                        allowed[
                            local_index
                        ]
                    ),
                    float(
                        subset_scores[
                            local_index
                        ]
                    ),
                )
                for local_index
                in local_order
            ]

        return [
            SearchHit(
                position=position,
                score=score,
                metadata=dict(
                    self.metadata[
                        position
                    ]
                ),
            )
            for position, score
            in ordered
        ]