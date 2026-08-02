from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from src.collection_config import ALLOWED_DOMAIN_SUFFIXES

REQUIRED_COLUMNS = (
    "source_id",
    "title",
    "url",
    "source_group",
    "audience",
    "degree_level",
    "faculty",
    "programme_name",
    "language",
    "is_external",
    "approved",
)

_TRUE_VALUES = {"true", "1", "yes", "y"}
_FALSE_VALUES = {"false", "0", "no", "n", ""}


def parse_bool(value: str, *, field_name: str, row_number: int) -> bool:
    normalised = (value or "").strip().lower()

    if normalised in _TRUE_VALUES:
        return True

    if normalised in _FALSE_VALUES:
        return False

    raise ValueError(
        f"Row {row_number}: {field_name} must be TRUE/FALSE, "
        f"but received {value!r}."
    )


def is_allowed_domain(hostname: str) -> bool:
    host = (hostname or "").lower().rstrip(".")

    return any(
        host == suffix or host.endswith(f".{suffix}")
        for suffix in ALLOWED_DOMAIN_SUFFIXES
    )


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    title: str
    url: str
    source_group: str
    audience: str
    degree_level: str
    faculty: str
    programme_name: str
    language: str
    is_external: bool
    approved: bool

    @property
    def hostname(self) -> str:
        return (urlparse(self.url).hostname or "").lower()

    def metadata(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "source_url": self.url,
            "source_group": self.source_group,
            "audience": self.audience,
            "degree_level": self.degree_level,
            "faculty": self.faculty,
            "programme_name": self.programme_name,
            "language": self.language,
            "is_external": self.is_external,
            "approved": self.approved,
        }


def load_sources(
    path: Path | str,
    *,
    approved_only: bool = True,
    enforce_allowed_domains: bool = True,
) -> list[SourceRecord]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Source catalogue not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in fieldnames
        ]

        if missing:
            raise ValueError(
                f"{path} is missing required columns: "
                f"{', '.join(missing)}"
            )

        records: list[SourceRecord] = []
        seen_ids: set[str] = set()
        seen_urls: set[str] = set()
        errors: list[str] = []

        for row_number, row in enumerate(reader, start=2):
            cleaned = {
                key: (row.get(key) or "").strip()
                for key in REQUIRED_COLUMNS
            }

            try:
                approved = parse_bool(
                    cleaned["approved"],
                    field_name="approved",
                    row_number=row_number,
                )

                is_external = parse_bool(
                    cleaned["is_external"],
                    field_name="is_external",
                    row_number=row_number,
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue

            source_id = cleaned["source_id"]
            url = cleaned["url"]
            parsed = urlparse(url)

            if not re.fullmatch(r"[A-Z0-9_]+", source_id):
                errors.append(
                    f"Row {row_number}: invalid source_id "
                    f"{source_id!r}; use uppercase letters, "
                    "digits and underscores only."
                )

            if source_id in seen_ids:
                errors.append(
                    f"Row {row_number}: duplicate source_id "
                    f"{source_id!r}."
                )

            if url in seen_urls:
                errors.append(
                    f"Row {row_number}: duplicate URL {url!r}."
                )

            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                errors.append(
                    f"Row {row_number}: invalid HTTP(S) URL {url!r}."
                )

            elif (
                enforce_allowed_domains
                and not is_allowed_domain(parsed.hostname)
            ):
                errors.append(
                    f"Row {row_number}: domain "
                    f"{parsed.hostname!r} is outside the "
                    "approved official-domain allow-list."
                )

            required_values = {
                "source_id": source_id,
                "title": cleaned["title"],
                "url": url,
                "source_group": cleaned["source_group"],
                "audience": cleaned["audience"],
                "degree_level": cleaned["degree_level"],
                "language": cleaned["language"],
            }

            for field_name, value in required_values.items():
                if not value:
                    errors.append(
                        f"Row {row_number}: {field_name} is empty."
                    )

            seen_ids.add(source_id)
            seen_urls.add(url)

            record = SourceRecord(
                source_id=source_id,
                title=cleaned["title"],
                url=url,
                source_group=cleaned["source_group"],
                audience=cleaned["audience"],
                degree_level=cleaned["degree_level"],
                faculty=cleaned["faculty"],
                programme_name=cleaned["programme_name"],
                language=cleaned["language"],
                is_external=is_external,
                approved=approved,
            )

            if not approved_only or approved:
                records.append(record)

    if errors:
        formatted = "\n".join(
            f"- {error}"
            for error in errors
        )

        raise ValueError(
            f"Catalogue validation failed:\n{formatted}"
        )

    return records