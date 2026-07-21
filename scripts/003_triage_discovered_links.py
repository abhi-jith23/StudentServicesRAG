from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collection_config import (  # noqa: E402
    DISCOVERED_LINKS_CSV,
    SOURCES_CSV,
)
from src.source_catalog import (  # noqa: E402
    is_allowed_domain,
    load_sources,
)


OUTPUT_FIELDS = (
    "url",
    "domain",
    "probable_type",
    "anchor_text",
    "parent_count",
    "score",
    "decision",
    "reason",
)

TRACKING_PARAMETERS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
    "_ga",
    "_gl",
}

REJECTED_EXTENSIONS = {
    ".7z",
    ".avi",
    ".css",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mov",
    ".mp3",
    ".mp4",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".rss",
    ".svg",
    ".tar",
    ".tif",
    ".tiff",
    ".webm",
    ".webp",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}

OUT_OF_SCOPE_PATTERNS = (
    r"/news(?:/|$)",
    r"/events?(?:/|$)",
    r"/research(?:/|$)",
    r"/doctoral(?:/|$)",
    r"/phd(?:/|$)",
    r"/people(?:/|$)",
    r"/staff(?:/|$)",
    r"/jobs?(?:/|$)",
    r"/vacanc(?:y|ies)(?:/|$)",
    r"/press(?:/|$)",
    r"/media(?:/|$)",
    r"/tag(?:/|$)",
    r"/author(?:/|$)",
    r"/feed(?:/|$)",
    r"/wp-json(?:/|$)",
)

HIGH_VALUE_TERMS = {
    "admission": 3,
    "admissions": 3,
    "application": 2,
    "apply": 2,
    "bachelor": 3,
    "master": 3,
    "programme": 2,
    "program": 2,
    "study programme": 3,
    "study program": 3,
    "required document": 3,
    "eligibility": 3,
    "language skill": 3,
    "language requirement": 3,
    "diploma recognition": 3,
    "reenrolment": 3,
    "re-enrolment": 3,
    "re enrolment": 3,
    "enrolment": 2,
    "enrollment": 2,
    "student status": 3,
    "study progression": 3,
    "academic conduct": 3,
    "appeal": 3,
    "complaint": 2,
    "special arrangement": 3,
    "exam": 2,
    "assessment": 2,
    "regulation": 3,
    "règlement": 3,
    "reglement": 3,
    "accommodation": 3,
    "housing": 3,
    "residence permit": 3,
    "authorisation to stay": 3,
    "authorization to stay": 3,
    "health insurance": 3,
    "tuition": 2,
    "fee": 2,
    "academic calendar": 3,
    "deadline": 2,
    "thesis": 2,
    "internship": 2,
}

LOW_VALUE_TERMS = (
    "privacy",
    "cookie",
    "newsletter",
    "social media",
    "facebook",
    "instagram",
    "linkedin",
    "youtube",
    "accessibility",
    "legal notice",
    "sitemap",
    "webmaster",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically deduplicate and triage "
            "discovered source links."
        )
    )

    parser.add_argument(
        "--links",
        type=Path,
        default=DISCOVERED_LINKS_CSV,
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=SOURCES_CSV,
    )

    parser.add_argument(
        "--all-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "catalog"
            / "discovered_links_triaged.csv"
        ),
    )

    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "catalog"
            / "candidate_link_backlog.csv"
        ),
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help=(
            "Maximum number of high-value links "
            "kept in the optional backlog."
        ),
    )

    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Discovered-link file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return [
            {
                key: (value or "").strip()
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in OUTPUT_FIELDS
                }
            )


def canonicalise_url(url: str) -> str:
    parsed = urlsplit(url.strip())

    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()

    if not scheme or not hostname:
        return ""

    port = parsed.port

    if port and not (
        (scheme == "https" and port == 443)
        or (scheme == "http" and port == 80)
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = re.sub(
        r"/{2,}",
        "/",
        parsed.path or "/",
    )

    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    filtered_query = [
        (key, value)
        for key, value in parse_qsl(
            parsed.query,
            keep_blank_values=True,
        )
        if key.lower() not in TRACKING_PARAMETERS
    ]

    query = urlencode(
        sorted(filtered_query),
        doseq=True,
    )

    return urlunsplit(
        (
            scheme,
            netloc,
            path,
            query,
            "",
        )
    )


def get_extension(url: str) -> str:
    path = urlsplit(url).path.lower()

    match = re.search(
        r"(\.[a-z0-9]{1,6})$",
        path,
    )

    return match.group(1) if match else ""


def is_out_of_scope(url: str) -> bool:
    path = urlsplit(url).path.lower()

    return any(
        re.search(pattern, path)
        for pattern in OUT_OF_SCOPE_PATTERNS
    )


def calculate_score(
    url: str,
    anchor_text: str,
    probable_type: str,
    parent_count: int,
) -> tuple[int, list[str]]:
    combined = (
        f"{url} {anchor_text}"
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    score = 0
    reasons: list[str] = []

    for term, weight in HIGH_VALUE_TERMS.items():
        if term in combined:
            score += weight
            reasons.append(
                f"matched:{term}"
            )

    for term in LOW_VALUE_TERMS:
        if term in combined:
            score -= 5
            reasons.append(
                f"low_value:{term}"
            )

    if probable_type == "pdf":
        score += 1
        reasons.append("official_pdf_candidate")

    if parent_count >= 3:
        score += 1
        reasons.append(
            "linked_from_multiple_sources"
        )

    return score, reasons


def main() -> int:
    args = parse_args()

    discovered = read_csv(args.links)

    catalogue = load_sources(
        args.catalog,
        approved_only=False,
    )

    catalogue_urls = {
        canonicalise_url(source.url)
        for source in catalogue
    }

    grouped: dict[
        str,
        dict[str, object],
    ] = {}

    for row in discovered:
        canonical_url = canonicalise_url(
            row.get("url", "")
        )

        if not canonical_url:
            continue

        entry = grouped.setdefault(
            canonical_url,
            {
                "url": canonical_url,
                "domain": (
                    urlsplit(
                        canonical_url
                    ).hostname
                    or ""
                ),
                "probable_type": (
                    row.get(
                        "probable_type",
                        ""
                    )
                    or (
                        "pdf"
                        if canonical_url
                        .lower()
                        .endswith(".pdf")
                        else "html"
                    )
                ),
                "anchors": set(),
                "parents": set(),
            },
        )

        anchor = row.get(
            "anchor_text",
            ""
        )

        parent = row.get(
            "parent_source_id",
            ""
        )

        if anchor:
            entry["anchors"].add(anchor)

        if parent:
            entry["parents"].add(parent)

    results: list[
        dict[str, object]
    ] = []

    for url, entry in grouped.items():
        domain = str(entry["domain"])
        probable_type = str(
            entry["probable_type"]
        )

        anchors = sorted(
            str(value)
            for value in entry["anchors"]
        )

        parents = {
            str(value)
            for value in entry["parents"]
        }

        anchor_text = " | ".join(
            anchors[:5]
        )

        parent_count = len(parents)
        extension = get_extension(url)

        decision = ""
        reason = ""
        score = 0

        if url in catalogue_urls:
            decision = "existing_source"
            reason = (
                "URL is already present "
                "in sources.csv."
            )

        elif not is_allowed_domain(domain):
            decision = "auto_reject"
            reason = (
                "Domain is outside the "
                "official allow-list."
            )

        elif extension in REJECTED_EXTENSIONS:
            decision = "auto_reject"
            reason = (
                f"Unsupported asset type: "
                f"{extension}"
            )

        elif is_out_of_scope(url):
            decision = "auto_reject"
            reason = (
                "URL matches an out-of-scope "
                "news, events, research, PhD, "
                "staff or recruitment path."
            )

        else:
            score, reasons = calculate_score(
                url=url,
                anchor_text=anchor_text,
                probable_type=probable_type,
                parent_count=parent_count,
            )

            if score >= 5:
                decision = "candidate_backlog"
                reason = (
                    "; ".join(reasons)
                    or "Relevant source terms."
                )

            else:
                decision = "low_priority"
                reason = (
                    "; ".join(reasons)
                    or "No strong relevance signal."
                )

        results.append(
            {
                "url": url,
                "domain": domain,
                "probable_type": probable_type,
                "anchor_text": anchor_text,
                "parent_count": parent_count,
                "score": score,
                "decision": decision,
                "reason": reason,
            }
        )

    priority_order = {
        "candidate_backlog": 0,
        "low_priority": 1,
        "existing_source": 2,
        "auto_reject": 3,
    }

    results.sort(
        key=lambda row: (
            priority_order.get(
                str(row["decision"]),
                99,
            ),
            -int(row["score"]),
            str(row["url"]),
        )
    )

    candidates = [
        row
        for row in results
        if row["decision"]
        == "candidate_backlog"
    ][: args.max_candidates]

    write_csv(
        args.all_output,
        results,
    )

    write_csv(
        args.candidate_output,
        candidates,
    )

    counts = Counter(
        str(row["decision"])
        for row in results
    )

    print(
        f"Raw discovered rows: "
        f"{len(discovered)}"
    )

    print(
        f"Unique canonical URLs: "
        f"{len(results)}"
    )

    print("\nAutomatic decisions:")

    for decision, count in sorted(
        counts.items()
    ):
        print(
            f"  {decision}: {count}"
        )

    print(
        f"\nOptional candidate backlog: "
        f"{len(candidates)}"
    )

    print(
        f"Full triage: "
        f"{args.all_output}"
    )

    print(
        f"Candidate backlog: "
        f"{args.candidate_output}"
    )

    print(
        "\nNo links were added to "
        "sources.csv automatically."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())