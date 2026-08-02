from __future__ import annotations

import re
import unicodedata
from dataclasses import (
    asdict,
    dataclass,
)
from typing import (
    Any,
    Iterable,
    Sequence,
)

import numpy as np

from src.config import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_MAX_CHUNKS_PER_SOURCE,
    DEFAULT_TOP_K,
)

from src.embedder import (
    E5Embedder,
)

from src.vectorstore import (
    FaissVectorStore,
    SearchHit,
)


def normalise_text(
    value: str,
) -> str:
    decomposed = unicodedata.normalize(
        "NFKD",
        value or "",
    )

    without_accents = "".join(
        character
        for character
        in decomposed
        if not unicodedata.combining(
            character
        )
    )

    lowered = (
        without_accents
        .casefold()
        .replace(
            "&",
            " and ",
        )
    )

    lowered = re.sub(
        r"[^a-z0-9]+",
        " ",
        lowered,
    )

    return re.sub(
        r"\s+",
        " ",
        lowered,
    ).strip()


@dataclass(
    frozen=True,
    slots=True,
)
class RouteDecision:
    programme_name: str = ""
    faculty: str = ""
    degree_level: str = ""
    audience: str = ""

    preferred_source_groups: (
        tuple[str, ...]
    ) = ()

    matched_aliases: (
        tuple[str, ...]
    ) = ()

    def to_dict(
        self,
    ) -> dict[str, Any]:
        payload = asdict(self)

        payload[
            "preferred_source_groups"
        ] = list(
            self.preferred_source_groups
        )

        payload[
            "matched_aliases"
        ] = list(
            self.matched_aliases
        )

        return payload


@dataclass(
    frozen=True,
    slots=True,
)
class RetrievalResult:
    rank: int
    position: int
    chunk_id: str
    source_id: str
    title: str
    source_url: str
    semantic_score: float
    adjusted_score: float
    source_group: str
    faculty: str
    programme_name: str
    degree_level: str
    audience: str
    is_external: bool
    section_path: str
    page_start: int | None
    page_end: int | None
    chunk_text: str

    matched_preferences: (
        tuple[str, ...]
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        payload = asdict(self)

        payload[
            "matched_preferences"
        ] = list(
            self.matched_preferences
        )

        return payload


class QueryRouter:
    FACULTY_ALIASES: (
        dict[str, str]
    ) = {
        "fstm": "FSTM",
        (
            "faculty of science "
            "technology and medicine"
        ): "FSTM",
        (
            "faculty science "
            "technology medicine"
        ): "FSTM",
        "fdef": "FDEF",
        (
            "faculty of law "
            "economics and finance"
        ): "FDEF",
        (
            "faculty law "
            "economics finance"
        ): "FDEF",
        "fhse": "FHSE",
        (
            "faculty of humanities "
            "education and social sciences"
        ): "FHSE",
        (
            "faculty humanities "
            "education social sciences"
        ): "FHSE",
    }

    TOPIC_KEYWORDS: dict[
        str,
        tuple[str, ...],
    ] = {
        "admissions": (
            "admission",
            "apply",
            "application",
            "eligibility",
            "entry requirement",
            "required document",
            "language proof",
            "proof of language",
            "proof of competence",
            "prove competence",
            "language certificate",
            "required language level",
            "meet the required language",
            "all teaching languages",
            "every teaching language",
            "language deadline",
        ),
        "programmes": (
            "programme",
            "programmes",
            "program",
            "programs",
            "study programme",
            "study programmes",
            "study program",
            "study programs",
            "compare programmes",
            "compare programs",
            "programme overview",
            "program overview",
            "programme catalogue",
            "program catalogue",
            "educational offer",
            "course structure",
            "curriculum",
            "ects",
            "duration",
            "career opportunity",
            "internship",
            "thesis",
        ),
        "reenrolment": (
            "re enrol",
            "reenrol",
            "re register",
            "next semester",
            "semester registration",
        ),
        "student_status": (
            "student status",
            "tuition payment",
            "semester fee",
            "leave programme",
            "change programme",
            "part time",
        ),
        "academic_matters": (
            "exam",
            "assessment",
            "study progression",
            "academic conduct",
            "plagiarism",
            "appeal",
            "complaint",
            "special arrangement",
            "reasonable adjustment",
            "extenuating circumstance",
        ),
        "regulations": (
            "regulation",
            "rule",
            "article",
            "legal basis",
            "maximum study duration",
        ),
        "accommodation": (
            "accommodation",
            "housing",
            "residence",
            "room",
            "lease",
            "rent",
            "tenant",
        ),
        "immigration": (
            "visa",
            "residence permit",
            "authorisation to stay",
            "authorization to stay",
            "third country",
            "non eu immigration",
        ),
        "health_insurance": (
            "health insurance",
            "cns",
            "ehic",
            (
                "european health "
                "insurance card"
            ),
        ),
        "financial_support": (
            "financial aid",
            "aidefi",
            "mengstudien",
            "cost of living",
            "budget",
            "scholarship",
        ),
        "diploma_recognition": (
            "diploma recognition",
            "recognition of diploma",
            "equivalence",
            "foreign diploma",
        ),
        "academic_calendar": (
            "academic calendar",
            "semester date",
            "exam period",
            "welcome week",
            "university holiday",
        ),
        "university_structure": (
            "faculty",
            "department",
            "university structure",
        ),
    }

    CURRENT_KEYWORDS = (
        "current student",
        "re enrol",
        "reenrol",
        "next semester",
        "exam",
        "appeal",
        "study progression",
        "student status",
        "lease",
    )

    PROSPECTIVE_KEYWORDS = (
        "prospective",
        "applicant",
        "apply",
        "application",
        "admission",
        "before enrolment",
    )

    def __init__(
        self,
        metadata: Sequence[
            dict[str, Any]
        ],
    ) -> None:
        self.metadata = list(
            metadata
        )

        self._programme_aliases = (
            self._build_programme_aliases()
        )

    def _build_programme_aliases(
        self,
    ) -> dict[
        str,
        set[
            tuple[str, str, str]
        ],
    ]:
        aliases: dict[
            str,
            set[
                tuple[str, str, str]
            ],
        ] = {}

        for row in self.metadata:
            programme_name = str(
                row.get(
                    "programme_name",
                    "",
                )
                or ""
            ).strip()

            if not programme_name:
                continue

            title = str(
                row.get(
                    "title",
                    "",
                )
                or ""
            ).strip()

            degree = str(
                row.get(
                    "degree_level",
                    "",
                )
                or ""
            ).strip()

            faculty = str(
                row.get(
                    "faculty",
                    "",
                )
                or ""
            ).strip()

            candidates = {
                normalise_text(
                    title
                ),
                normalise_text(
                    programme_name.replace(
                        "_",
                        " ",
                    )
                ),
            }

            title_without_degree = (
                re.sub(
                    (
                        r"^(?:bachelor|master)"
                        r"(?:\s+in|\s+en)?\s+"
                    ),
                    "",
                    normalise_text(
                        title
                    ),
                )
            )

            if title_without_degree:
                candidates.add(
                    title_without_degree
                )

            for alias in candidates:
                if len(alias) < 3:
                    continue

                aliases.setdefault(
                    alias,
                    set(),
                ).add(
                    (
                        programme_name,
                        degree,
                        faculty,
                    )
                )

        return aliases

    @staticmethod
    def _detect_degree(
        query: str,
    ) -> str:
        bachelor = bool(
            re.search(
                (
                    r"\b(?:bachelor|bsc|"
                    r"undergraduate)\b"
                ),
                query,
            )
        )

        master = bool(
            re.search(
                (
                    r"\b(?:master|masters|"
                    r"msc|graduate)\b"
                ),
                query,
            )
        )

        if bachelor and master:
            return "bachelor_master"

        if bachelor:
            return "bachelor"

        if master:
            return "master"

        return ""

    def _detect_faculty(
        self,
        query: str,
    ) -> tuple[
        str,
        list[str],
    ]:
        matches: list[
            tuple[int, str, str]
        ] = []

        for alias, faculty in (
            self.FACULTY_ALIASES.items()
        ):
            if re.search(
                rf"\b{re.escape(alias)}\b",
                query,
            ):
                matches.append(
                    (
                        len(alias),
                        alias,
                        faculty,
                    )
                )

        if not matches:
            return "", []

        matches.sort(
            reverse=True
        )

        return (
            matches[0][2],
            [matches[0][1]],
        )

    def _detect_programme(
        self,
        query: str,
        *,
        degree_level: str,
        faculty: str,
    ) -> tuple[
        str,
        str,
        str,
        list[str],
    ]:
        matches: list[
            tuple[
                int,
                str,
                set[
                    tuple[
                        str,
                        str,
                        str,
                    ]
                ],
            ]
        ] = []

        for alias, candidates in (
            self._programme_aliases.items()
        ):
            if re.search(
                rf"\b{re.escape(alias)}\b",
                query,
            ):
                matches.append(
                    (
                        len(alias),
                        alias,
                        candidates,
                    )
                )

        if not matches:
            return (
                "",
                degree_level,
                faculty,
                [],
            )

        matches.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        for (
            _,
            alias,
            raw_candidates,
        ) in matches:
            candidates = list(
                raw_candidates
            )

            if degree_level in {
                "bachelor",
                "master",
            }:
                narrowed = [
                    item
                    for item
                    in candidates
                    if (
                        item[1]
                        == degree_level
                    )
                ]

                if narrowed:
                    candidates = narrowed

            if faculty:
                narrowed = [
                    item
                    for item
                    in candidates
                    if item[2] == faculty
                ]

                if narrowed:
                    candidates = narrowed

            programme_names = {
                item[0]
                for item
                in candidates
            }

            if (
                len(programme_names)
                == 1
            ):
                programme_name = next(
                    iter(
                        programme_names
                    )
                )

                resolved_degree = (
                    degree_level
                )

                resolved_faculty = (
                    faculty
                )

                if len(candidates) == 1:
                    resolved_degree = (
                        resolved_degree
                        or candidates[0][1]
                    )

                    resolved_faculty = (
                        resolved_faculty
                        or candidates[0][2]
                    )

                return (
                    programme_name,
                    resolved_degree,
                    resolved_faculty,
                    [alias],
                )

        return (
            "",
            degree_level,
            faculty,
            [],
        )

    def _detect_source_groups(
        self,
        query: str,
    ) -> tuple[
        tuple[str, ...],
        list[str],
    ]:
        scored: list[
            tuple[
                int,
                str,
                list[str],
            ]
        ] = []

        for group, keywords in (
            self.TOPIC_KEYWORDS.items()
        ):
            matched = [
                keyword
                for keyword
                in keywords
                if re.search(
                    (
                        rf"\b"
                        f"{re.escape(normalise_text(keyword))}"
                        rf"\b"
                    ),
                    query,
                )
            ]

            if matched:
                score = sum(
                    max(
                        1,
                        len(
                            keyword.split()
                        ),
                    )
                    for keyword
                    in matched
                )

                scored.append(
                    (
                        score,
                        group,
                        matched,
                    )
                )

        if not scored:
            return (), []

        scored.sort(
            reverse=True
        )

        best_score = scored[0][0]

        selected = [
            group
            for score, group, _
            in scored
            if (
                score
                >= best_score - 1
            )
        ]

        aliases = [
            keyword
            for score, _, items
            in scored
            if (
                score
                >= best_score - 1
            )
            for keyword
            in items
        ]

        return (
            tuple(
                dict.fromkeys(
                    selected
                )
            ),
            aliases,
        )

    def _detect_audience(
        self,
        query: str,
    ) -> tuple[
        str,
        list[str],
    ]:
        current_matches = [
            keyword
            for keyword
            in self.CURRENT_KEYWORDS
            if keyword in query
        ]

        prospective_matches = [
            keyword
            for keyword
            in self.PROSPECTIVE_KEYWORDS
            if keyword in query
        ]

        if (
            current_matches
            and not prospective_matches
        ):
            return (
                "current",
                current_matches,
            )

        if (
            prospective_matches
            and not current_matches
        ):
            return (
                "prospective",
                prospective_matches,
            )

        return (
            "",
            (
                current_matches
                + prospective_matches
            ),
        )

    def route(
        self,
        question: str,
    ) -> RouteDecision:
        query = normalise_text(
            question
        )

        degree_level = (
            self._detect_degree(
                query
            )
        )

        (
            faculty,
            faculty_aliases,
        ) = self._detect_faculty(
            query
        )

        (
            programme_name,
            degree_level,
            faculty,
            programme_aliases,
        ) = self._detect_programme(
            query,
            degree_level=(
                degree_level
            ),
            faculty=faculty,
        )

        (
            source_groups,
            topic_aliases,
        ) = self._detect_source_groups(
            query
        )

        (
            audience,
            audience_aliases,
        ) = self._detect_audience(
            query
        )

        return RouteDecision(
            programme_name=(
                programme_name
            ),
            faculty=faculty,
            degree_level=(
                degree_level
            ),
            audience=audience,
            preferred_source_groups=(
                source_groups
            ),
            matched_aliases=tuple(
                dict.fromkeys(
                    programme_aliases
                    + faculty_aliases
                    + topic_aliases
                    + audience_aliases
                )
            ),
        )


class Retriever:
    def __init__(
        self,
        *,
        store: FaissVectorStore,
        embedder: E5Embedder,
        candidate_k: int = (
            DEFAULT_CANDIDATE_K
        ),
        max_chunks_per_source: (
            int
        ) = (
            DEFAULT_MAX_CHUNKS_PER_SOURCE
        ),
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.candidate_k = (
            candidate_k
        )

        self.max_chunks_per_source = (
            max_chunks_per_source
        )

        self.router = QueryRouter(
            store.metadata
        )

    @staticmethod
    def _compatible_degree(
        row_degree: str,
        wanted: str,
    ) -> bool:
        if (
            not wanted
            or wanted
            == "bachelor_master"
        ):
            return True

        return row_degree in {
            wanted,
            "bachelor_master",
        }

    @staticmethod
    def _compatible_audience(
        row_audience: str,
        wanted: str,
    ) -> bool:
        if not wanted:
            return True

        if wanted == "prospective":
            return row_audience in {
                "prospective",
                "prospective_current",
                "both",
            }

        if wanted == "current":
            return row_audience in {
                "current",
                "prospective_current",
                "both",
            }

        return True

    def _targeted_positions(
        self,
        route: RouteDecision,
    ) -> np.ndarray | None:
        if route.programme_name:
            return (
                self.store
                .positions_matching(
                    lambda row: (
                        str(
                            row.get(
                                "programme_name",
                                "",
                            )
                        )
                        == route.programme_name
                    )
                    and (
                        self._compatible_degree(
                            str(
                                row.get(
                                    "degree_level",
                                    "",
                                )
                            ),
                            route.degree_level,
                        )
                    )
                )
            )

        if route.faculty:
            return (
                self.store
                .positions_matching(
                    lambda row: (
                        str(
                            row.get(
                                "faculty",
                                "",
                            )
                        )
                        == route.faculty
                    )
                    and (
                        self._compatible_degree(
                            str(
                                row.get(
                                    "degree_level",
                                    "",
                                )
                            ),
                            route.degree_level,
                        )
                    )
                )
            )

        return None

    def _general_positions(
        self,
        route: RouteDecision,
    ) -> np.ndarray:
        return (
            self.store
            .positions_matching(
                lambda row: (
                    not str(
                        row.get(
                            "programme_name",
                            "",
                        )
                        or ""
                    ).strip()
                )
                and (
                    self._compatible_degree(
                        str(
                            row.get(
                                "degree_level",
                                "",
                            )
                        ),
                        route.degree_level,
                    )
                )
                and (
                    self._compatible_audience(
                        str(
                            row.get(
                                "audience",
                                "",
                            )
                        ),
                        route.audience,
                    )
                )
            )
        )

    @staticmethod
    def _merge_hits(
        hit_groups: Iterable[
            Sequence[SearchHit]
        ],
    ) -> list[SearchHit]:
        best: dict[
            int,
            SearchHit,
        ] = {}

        for hits in hit_groups:
            for hit in hits:
                previous = best.get(
                    hit.position
                )

                if (
                    previous is None
                    or (
                        hit.score
                        > previous.score
                    )
                ):
                    best[
                        hit.position
                    ] = hit

        return list(
            best.values()
        )

    def _adjust_score(
        self,
        hit: SearchHit,
        route: RouteDecision,
    ) -> tuple[
        float,
        tuple[str, ...],
    ]:
        row = hit.metadata
        score = float(
            hit.score
        )

        matched: list[str] = []

        if (
            route.programme_name
            and (
                row.get(
                    "programme_name"
                )
                == route.programme_name
            )
        ):
            score += 0.080
            matched.append(
                "programme"
            )

        if (
            route.faculty
            and (
                row.get(
                    "faculty"
                )
                == route.faculty
            )
        ):
            score += 0.035
            matched.append(
                "faculty"
            )

        if (
            route.degree_level
            and (
                self._compatible_degree(
                    str(
                        row.get(
                            "degree_level",
                            "",
                        )
                    ),
                    route.degree_level,
                )
            )
        ):
            score += 0.025
            matched.append(
                "degree_level"
            )

        if (
            route.audience
            and (
                self._compatible_audience(
                    str(
                        row.get(
                            "audience",
                            "",
                        )
                    ),
                    route.audience,
                )
            )
        ):
            score += 0.015
            matched.append(
                "audience"
            )

        source_group = str(
            row.get(
                "source_group",
                "",
            )
        )

        if (
            route.preferred_source_groups
        ):
            if (
                source_group
                in (
                    route
                    .preferred_source_groups
                )
            ):
                score += 0.045
                matched.append(
                    "source_group"
                )

            elif (
                source_group
                == "regulations"
                and any(
                    group in {
                        "academic_matters",
                        "reenrolment",
                        "student_status",
                        "admissions",
                    }
                    for group
                    in (
                        route
                        .preferred_source_groups
                    )
                )
            ):
                score += 0.015
                matched.append(
                    "regulations_support"
                )

        if (
            not route.programme_name
            and (
                "programmes"
                in route.preferred_source_groups
            )
            and source_group == "programmes"
            and not str(
                row.get(
                    "programme_name",
                    "",
                )
                or ""
            ).strip()
        ):
            score += 0.060

            matched.append(
                "programme_catalogue"
            )

            if not str(
                row.get(
                    "faculty",
                    "",
                )
                or ""
            ).strip():
                score += 0.040

                matched.append(
                    "university_wide_catalogue"
                )

        if (
            bool(
                row.get(
                    "is_external",
                    False,
                )
            )
            and any(
                group in {
                    "immigration",
                    "health_insurance",
                    "financial_support",
                    "diploma_recognition",
                }
                for group
                in (
                    route
                    .preferred_source_groups
                )
            )
        ):
            score += 0.010
            matched.append(
                "official_external"
            )

        return (
            score,
            tuple(matched),
        )

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = (
            DEFAULT_TOP_K
        ),
    ) -> tuple[
        RouteDecision,
        list[RetrievalResult],
    ]:
        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if top_k <= 0:
            return (
                self.router.route(
                    question
                ),
                [],
            )

        route = self.router.route(
            question
        )

        query_vector = (
            self.embedder
            .encode_queries(
                [question],
                show_progress_bar=False,
            )
        )

        global_hits = (
            self.store.search(
                query_vector,
                top_k=min(
                    self.candidate_k,
                    len(
                        self.store.metadata
                    ),
                ),
            )
        )

        hit_groups: list[
            Sequence[SearchHit]
        ] = [
            global_hits
        ]

        targeted_positions = (
            self._targeted_positions(
                route
            )
        )

        if (
            targeted_positions
            is not None
            and targeted_positions.size
        ):
            hit_groups.append(
                self.store.search(
                    query_vector,
                    top_k=min(
                        (
                            self.candidate_k
                            // 2
                        ),
                        int(
                            targeted_positions
                            .size
                        ),
                    ),
                    allowed_positions=(
                        targeted_positions
                    ),
                )
            )

        if (
            route.programme_name
            or route.faculty
            or route.degree_level
            or route.audience
        ):
            general_positions = (
                self._general_positions(
                    route
                )
            )

            if general_positions.size:
                hit_groups.append(
                    self.store.search(
                        query_vector,
                        top_k=min(
                            (
                                self.candidate_k
                                // 2
                            ),
                            int(
                                general_positions
                                .size
                            ),
                        ),
                        allowed_positions=(
                            general_positions
                        ),
                    )
                )

        merged = self._merge_hits(
            hit_groups
        )

        scored: list[
            tuple[
                float,
                SearchHit,
                tuple[str, ...],
            ]
        ] = []

        for hit in merged:
            adjusted, matched = (
                self._adjust_score(
                    hit,
                    route,
                )
            )

            scored.append(
                (
                    adjusted,
                    hit,
                    matched,
                )
            )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1].score,
            ),
            reverse=True,
        )

        source_counts: dict[
            str,
            int,
        ] = {}

        selected: list[
            tuple[
                float,
                SearchHit,
                tuple[str, ...],
            ]
        ] = []

        for item in scored:
            _, hit, _ = item

            source_id = str(
                hit.metadata.get(
                    "source_id",
                    "",
                )
            )

            if (
                source_counts.get(
                    source_id,
                    0,
                )
                >= (
                    self
                    .max_chunks_per_source
                )
            ):
                continue

            selected.append(item)

            source_counts[
                source_id
            ] = (
                source_counts.get(
                    source_id,
                    0,
                )
                + 1
            )

            if (
                len(selected)
                >= top_k
            ):
                break

        results: list[
            RetrievalResult
        ] = []

        for rank, (
            adjusted_score,
            hit,
            matched,
        ) in enumerate(
            selected,
            start=1,
        ):
            row = hit.metadata

            results.append(
                RetrievalResult(
                    rank=rank,
                    position=(
                        hit.position
                    ),
                    chunk_id=str(
                        row.get(
                            "chunk_id",
                            "",
                        )
                    ),
                    source_id=str(
                        row.get(
                            "source_id",
                            "",
                        )
                    ),
                    title=str(
                        row.get(
                            "title",
                            "",
                        )
                    ),
                    source_url=str(
                        row.get(
                            "source_url",
                            "",
                        )
                    ),
                    semantic_score=float(
                        hit.score
                    ),
                    adjusted_score=float(
                        adjusted_score
                    ),
                    source_group=str(
                        row.get(
                            "source_group",
                            "",
                        )
                    ),
                    faculty=str(
                        row.get(
                            "faculty",
                            "",
                        )
                        or ""
                    ),
                    programme_name=str(
                        row.get(
                            "programme_name",
                            "",
                        )
                        or ""
                    ),
                    degree_level=str(
                        row.get(
                            "degree_level",
                            "",
                        )
                        or ""
                    ),
                    audience=str(
                        row.get(
                            "audience",
                            "",
                        )
                        or ""
                    ),
                    is_external=bool(
                        row.get(
                            "is_external",
                            False,
                        )
                    ),
                    section_path=str(
                        row.get(
                            "section_path",
                            "",
                        )
                        or ""
                    ),
                    page_start=row.get(
                        "page_start"
                    ),
                    page_end=row.get(
                        "page_end"
                    ),
                    chunk_text=str(
                        row.get(
                            "chunk_text",
                            "",
                        )
                    ),
                    matched_preferences=(
                        matched
                    ),
                )
            )

        return route, results