from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.cleaner import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_text,
    build_markdown_document,
    sha256_bytes,
    utc_now_iso,
)
from src.collection_config import (  # noqa: E402
    CLEAN_HTML_DIR,
    CLEAN_PDF_DIR,
    COLLECTION_ISSUES_CSV,
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_DELAY_SECONDS,
    DEFAULT_MAX_DOWNLOAD_BYTES,
    DEFAULT_READ_TIMEOUT_SECONDS,
    DISCOVERED_LINKS_CSV,
    FETCH_MANIFEST_CSV,
    HTML_MIN_TEXT_LENGTH,
    RAW_HTML_DIR,
    RAW_PDF_DIR,
    SOURCES_CSV,
)
from src.html_loader import (  # noqa: E402
    BrowserHTMLFetcher,
    decode_html,
    discover_links,
    extract_html,
)
from src.pdf_loader import extract_pdf  # noqa: E402
from src.source_catalog import (  # noqa: E402
    SourceRecord,
    load_sources,
)

USER_AGENT = (
    "UniLuStudentServicesRAG/0.1 "
    "(academic prototype; controlled "
    "official-source collection)"
)

MANIFEST_FIELDS = (
    "source_id",
    "original_url",
    "final_url",
    "detected_type",
    "http_status",
    "content_type",
    "fetch_method",
    "local_raw_path",
    "local_clean_path",
    "sha256",
    "fetched_at",
    "extraction_status",
    "text_length",
    "page_count",
    "error",
)

DISCOVERED_FIELDS = (
    "parent_source_id",
    "url",
    "anchor_text",
    "domain",
    "probable_type",
    "approved",
    "reason",
    "discovered_at",
)

ISSUE_FIELDS = (
    "source_id",
    "issue_type",
    "description",
    "resolution",
    "status",
    "recorded_at",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect, cache and clean approved "
            "Uni.lu corpus sources."
        )
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=SOURCES_CSV,
        help="Path to sources.csv.",
    )

    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help=(
            "Collect only this source ID. "
            "Repeat for multiple IDs."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Collect at most this many sources "
            "after filtering."
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Re-fetch sources even when cached "
            "files and a manifest entry exist."
        ),
    )

    parser.add_argument(
        "--headed",
        action="store_true",
        help=(
            "Run Playwright visibly when browser "
            "fallback is required."
        ),
    )

    parser.add_argument(
        "--no-playwright",
        action="store_true",
        help="Disable Playwright fallback.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help=(
            "Delay between network requests "
            "in seconds."
        ),
    )

    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_DOWNLOAD_BYTES,
        help=(
            "Maximum allowed response size "
            "in bytes."
        ),
    )

    return parser.parse_args()


def ensure_directories() -> None:
    for directory in (
        RAW_HTML_DIR,
        RAW_PDF_DIR,
        CLEAN_HTML_DIR,
        CLEAN_PDF_DIR,
        FETCH_MANIFEST_CSV.parent,
        DISCOVERED_LINKS_CSV.parent,
        COLLECTION_ISSUES_CSV.parent,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    if (
        not path.exists()
        or path.stat().st_size == 0
    ):
        return []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return [
            {
                key: (value or "")
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def write_csv_rows(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    fields = list(fieldnames)

    with temp_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fields
                }
            )

    temp_path.replace(path)


def upsert_row(
    rows: list[dict[str, str]],
    new_row: dict[str, object],
    *,
    key_fields: tuple[str, ...],
) -> None:
    target_key = tuple(
        str(new_row.get(field, ""))
        for field in key_fields
    )

    for index, existing in enumerate(rows):
        existing_key = tuple(
            str(existing.get(field, ""))
            for field in key_fields
        )

        if existing_key == target_key:
            rows[index] = {
                key: str(value)
                for key, value in new_row.items()
            }
            return

    rows.append(
        {
            key: str(value)
            for key, value in new_row.items()
        }
    )


def make_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": (
                "text/html,"
                "application/xhtml+xml,"
                "application/pdf;"
                "q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": (
                "en-GB,en;q=0.9"
            ),
        }
    )

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    return session


def fetch_bytes(
    session: requests.Session,
    url: str,
    *,
    max_bytes: int,
) -> tuple[bytes, str, int, str]:
    timeout = (
        DEFAULT_CONNECT_TIMEOUT_SECONDS,
        DEFAULT_READ_TIMEOUT_SECONDS,
    )

    with session.get(
        url,
        timeout=timeout,
        allow_redirects=True,
        stream=True,
    ) as response:
        response.raise_for_status()

        declared_length = response.headers.get(
            "content-length"
        )

        if declared_length:
            try:
                parsed_length = int(
                    declared_length
                )

                if parsed_length > max_bytes:
                    raise ValueError(
                        f"Content-Length "
                        f"{declared_length} exceeds "
                        f"the configured limit of "
                        f"{max_bytes} bytes."
                    )

            except ValueError as exc:
                if "exceeds" in str(exc):
                    raise

        chunks: list[bytes] = []
        downloaded = 0

        for chunk in response.iter_content(
            chunk_size=64 * 1024
        ):
            if not chunk:
                continue

            downloaded += len(chunk)

            if downloaded > max_bytes:
                raise ValueError(
                    "Downloaded response exceeded "
                    f"{max_bytes} bytes."
                )

            chunks.append(chunk)

        return (
            b"".join(chunks),
            response.url,
            response.status_code,
            response.headers.get(
                "content-type",
                "",
            ),
        )


def detect_source_type(
    *,
    content_type: str,
    data: bytes,
    final_url: str,
) -> str:
    media_type = (
        content_type
        or ""
    ).split(
        ";",
        1,
    )[0].strip().lower()

    sample = (
        data[:4096]
        .lstrip()
        .lower()
    )

    clean_url = (
        final_url
        .lower()
        .split("?", 1)[0]
    )

    if (
        media_type == "application/pdf"
        or data.startswith(b"%PDF-")
    ):
        return "pdf"

    if media_type in {
        "text/html",
        "application/xhtml+xml",
    }:
        return "html"

    if (
        sample.startswith(b"<!doctype html")
        or b"<html" in sample
    ):
        return "html"

    if clean_url.endswith(".pdf"):
        return "pdf"

    return "unsupported"


def is_cached(
    source: SourceRecord,
    manifest_by_id: dict[
        str,
        dict[str, str],
    ],
) -> bool:
    entry = manifest_by_id.get(
        source.source_id
    )

    if not entry:
        return False

    raw = PROJECT_ROOT / entry.get(
        "local_raw_path",
        "",
    )

    clean = PROJECT_ROOT / entry.get(
        "local_clean_path",
        "",
    )

    status = entry.get(
        "extraction_status",
        "",
    )

    return (
        raw.is_file()
        and clean.is_file()
        and status not in {
            "fetch_error",
            "unsupported_type",
            "blocked_page",
            "extraction_error",
        }
    )


def relative(
    path: Path,
) -> str:
    return str(
        path.resolve().relative_to(
            PROJECT_ROOT.resolve()
        )
    )


def issue_row(
    source_id: str,
    issue_type: str,
    description: str,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "issue_type": issue_type,
        "description": description,
        "resolution": "",
        "status": "open",
        "recorded_at": utc_now_iso(),
    }


def collect_html(
    *,
    source: SourceRecord,
    initial_data: bytes,
    initial_final_url: str,
    initial_status: int,
    initial_content_type: str,
    browser: BrowserHTMLFetcher | None,
) -> tuple[
    dict[str, object],
    list[dict[str, str]],
    list[dict[str, object]],
]:
    issues: list[dict[str, object]] = []

    html = decode_html(initial_data)
    final_url = initial_final_url
    http_status = initial_status
    content_type = initial_content_type
    fetch_method = "requests"

    extraction = extract_html(
        html,
        url=final_url,
    )

    too_short = (
        len(extraction.markdown)
        < HTML_MIN_TEXT_LENGTH
    )

    browser_needed = (
        extraction.block_page_suspected
        or too_short
    )

    if (
        browser_needed
        and browser is not None
    ):
        try:
            rendered = browser.fetch(
                source.url
            )

            rendered_extraction = extract_html(
                rendered.html,
                url=rendered.final_url,
            )

            if (
                len(rendered_extraction.markdown)
                > len(extraction.markdown)
            ):
                html = rendered.html
                final_url = rendered.final_url

                http_status = (
                    rendered.http_status
                    or http_status
                )

                content_type = (
                    rendered.content_type
                    or content_type
                )

                extraction = (
                    rendered_extraction
                )

                fetch_method = "playwright"

        except Exception as exc:
            issues.append(
                issue_row(
                    source.source_id,
                    "playwright_fallback_failed",
                    "Browser fallback failed; "
                    "the Requests result was retained "
                    f"({exc}).",
                )
            )

    raw_path = (
        RAW_HTML_DIR
        / f"{source.source_id}.html"
    )

    clean_path = (
        CLEAN_HTML_DIR
        / f"{source.source_id}.md"
    )

    if extraction.block_page_suspected:
        status = "blocked_page"

        issues.append(
            issue_row(
                source.source_id,
                "blocked_page",
                "The fetched HTML looks like "
                "a robot-verification or "
                "access-denied page.",
            )
        )

    elif (
        len(extraction.markdown)
        < HTML_MIN_TEXT_LENGTH
    ):
        status = "needs_manual_review"

        issues.append(
            issue_row(
                source.source_id,
                "short_html_extraction",
                f"Only "
                f"{len(extraction.markdown)} "
                "characters were extracted.",
            )
        )

    else:
        status = "extraction_ok"

    if extraction.used_fallback_extractor:
        issues.append(
            issue_row(
                source.source_id,
                "fallback_html_extractor_used",
                "Trafilatura returned too little "
                "text, so the BeautifulSoup "
                "fallback was used.",
            )
        )

        if status == "extraction_ok":
            status = "needs_manual_review"

    if status == "blocked_page":
        raise ValueError(
            "A verification/access-denied "
            "page was returned instead of "
            "usable content."
        )

    metadata = source.metadata()

    metadata.update(
        {
            "final_url": final_url,
            "detected_type": "html",
            "retrieved_at": utc_now_iso(),
            "page_title": extraction.page_title,
            "fetch_method": fetch_method,
        }
    )

    markdown_document = (
        build_markdown_document(
            metadata,
            extraction.markdown,
        )
    )

    atomic_write_text(
        raw_path,
        html,
    )

    atomic_write_text(
        clean_path,
        markdown_document,
    )

    links = discover_links(
        html,
        base_url=final_url,
    )

    manifest = {
        "source_id": source.source_id,
        "original_url": source.url,
        "final_url": final_url,
        "detected_type": "html",
        "http_status": http_status,
        "content_type": content_type,
        "fetch_method": fetch_method,
        "local_raw_path": relative(
            raw_path
        ),
        "local_clean_path": relative(
            clean_path
        ),
        "sha256": sha256_bytes(
            html.encode("utf-8")
        ),
        "fetched_at": utc_now_iso(),
        "extraction_status": status,
        "text_length": len(
            extraction.markdown
        ),
        "page_count": "",
        "error": "",
    }

    return manifest, links, issues


def collect_pdf(
    *,
    source: SourceRecord,
    data: bytes,
    final_url: str,
    http_status: int,
    content_type: str,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    raw_path = (
        RAW_PDF_DIR
        / f"{source.source_id}.pdf"
    )

    clean_path = (
        CLEAN_PDF_DIR
        / f"{source.source_id}.md"
    )

    atomic_write_bytes(
        raw_path,
        data,
    )

    extraction = extract_pdf(
        raw_path
    )

    metadata = source.metadata()

    metadata.update(
        {
            "final_url": final_url,
            "detected_type": "pdf",
            "retrieved_at": utc_now_iso(),
            "fetch_method": "requests",
            "page_count": (
                extraction.page_count
            ),
        }
    )

    markdown_document = (
        build_markdown_document(
            metadata,
            extraction.markdown,
        )
    )

    atomic_write_text(
        clean_path,
        markdown_document,
    )

    issues = [
        issue_row(
            source.source_id,
            "pdf_extraction_warning",
            description,
        )
        for description
        in extraction.issues
    ]

    manifest = {
        "source_id": source.source_id,
        "original_url": source.url,
        "final_url": final_url,
        "detected_type": "pdf",
        "http_status": http_status,
        "content_type": content_type,
        "fetch_method": "requests",
        "local_raw_path": relative(
            raw_path
        ),
        "local_clean_path": relative(
            clean_path
        ),
        "sha256": sha256_bytes(data),
        "fetched_at": utc_now_iso(),
        "extraction_status": (
            extraction.extraction_status
        ),
        "text_length": (
            extraction.text_length
        ),
        "page_count": (
            extraction.page_count
        ),
        "error": "",
    }

    return manifest, issues


def main() -> int:
    args = parse_args()
    ensure_directories()

    sources = load_sources(
        args.catalog,
        approved_only=True,
    )

    if args.source_id:
        requested = set(
            args.source_id
        )

        known = {
            source.source_id
            for source in sources
        }

        unknown = sorted(
            requested - known
        )

        if unknown:
            raise ValueError(
                "Unknown or unapproved "
                "source IDs: "
                f"{', '.join(unknown)}"
            )

        sources = [
            source
            for source in sources
            if source.source_id in requested
        ]

    if args.limit is not None:
        sources = sources[: args.limit]

    manifest_rows = read_csv_rows(
        FETCH_MANIFEST_CSV
    )

    discovered_rows = read_csv_rows(
        DISCOVERED_LINKS_CSV
    )

    issue_rows = read_csv_rows(
        COLLECTION_ISSUES_CSV
    )

    manifest_by_id = {
        row.get("source_id", ""): row
        for row in manifest_rows
    }

    catalogue_urls = {
        source.url
        for source in load_sources(
            args.catalog,
            approved_only=False,
        )
    }

    session = make_session()

    browser: BrowserHTMLFetcher | None = None

    collected = 0
    skipped = 0
    failed = 0

    try:
        for position, source in enumerate(
            sources,
            start=1,
        ):
            print(
                f"[{position}/{len(sources)}] "
                f"{source.source_id}: "
                f"{source.title}"
            )

            if (
                not args.refresh
                and is_cached(
                    source,
                    manifest_by_id,
                )
            ):
                print(
                    "  cached: skipped"
                )

                skipped += 1
                continue

            try:
                (
                    data,
                    final_url,
                    http_status,
                    content_type,
                ) = fetch_bytes(
                    session,
                    source.url,
                    max_bytes=args.max_bytes,
                )

                detected_type = detect_source_type(
                    content_type=content_type,
                    data=data,
                    final_url=final_url,
                )

                if detected_type == "html":
                    if (
                        browser is None
                        and not args.no_playwright
                    ):
                        browser = (
                            BrowserHTMLFetcher(
                                headed=args.headed,
                                user_agent=USER_AGENT,
                            )
                        )

                    (
                        manifest,
                        links,
                        new_issues,
                    ) = collect_html(
                        source=source,
                        initial_data=data,
                        initial_final_url=final_url,
                        initial_status=http_status,
                        initial_content_type=content_type,
                        browser=browser,
                    )

                    for link in links:
                        if (
                            link["url"]
                            in catalogue_urls
                        ):
                            continue

                        discovered = {
                            "parent_source_id": (
                                source.source_id
                            ),
                            **link,
                            "approved": "FALSE",
                            "reason": "",
                            "discovered_at": (
                                utc_now_iso()
                            ),
                        }

                        upsert_row(
                            discovered_rows,
                            discovered,
                            key_fields=(
                                "parent_source_id",
                                "url",
                            ),
                        )

                elif detected_type == "pdf":
                    (
                        manifest,
                        new_issues,
                    ) = collect_pdf(
                        source=source,
                        data=data,
                        final_url=final_url,
                        http_status=http_status,
                        content_type=content_type,
                    )

                else:
                    raise ValueError(
                        "Unsupported response type. "
                        f"Content-Type="
                        f"{content_type!r}, "
                        f"final URL={final_url!r}."
                    )

                upsert_row(
                    manifest_rows,
                    manifest,
                    key_fields=(
                        "source_id",
                    ),
                )

                manifest_by_id[
                    source.source_id
                ] = {
                    key: str(value)
                    for key, value
                    in manifest.items()
                }

                for issue in new_issues:
                    upsert_row(
                        issue_rows,
                        issue,
                        key_fields=(
                            "source_id",
                            "issue_type",
                            "description",
                        ),
                    )

                collected += 1

                print(
                    f"  "
                    f"{manifest['detected_type']} "
                    f"-> "
                    f"{manifest['extraction_status']} "
                    f"("
                    f"{manifest['text_length']} "
                    f"chars)"
                )

            except Exception as exc:
                failed += 1

                message = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    f"  ERROR: {message}"
                )

                error_manifest = {
                    "source_id": (
                        source.source_id
                    ),
                    "original_url": (
                        source.url
                    ),
                    "final_url": "",
                    "detected_type": "",
                    "http_status": "",
                    "content_type": "",
                    "fetch_method": "",
                    "local_raw_path": "",
                    "local_clean_path": "",
                    "sha256": "",
                    "fetched_at": (
                        utc_now_iso()
                    ),
                    "extraction_status": (
                        "fetch_error"
                    ),
                    "text_length": "",
                    "page_count": "",
                    "error": message,
                }

                upsert_row(
                    manifest_rows,
                    error_manifest,
                    key_fields=(
                        "source_id",
                    ),
                )

                manifest_by_id[
                    source.source_id
                ] = {
                    key: str(value)
                    for key, value
                    in error_manifest.items()
                }

                upsert_row(
                    issue_rows,
                    issue_row(
                        source.source_id,
                        "collection_error",
                        message,
                    ),
                    key_fields=(
                        "source_id",
                        "issue_type",
                        "description",
                    ),
                )

            write_csv_rows(
                FETCH_MANIFEST_CSV,
                MANIFEST_FIELDS,
                manifest_rows,
            )

            write_csv_rows(
                DISCOVERED_LINKS_CSV,
                DISCOVERED_FIELDS,
                discovered_rows,
            )

            write_csv_rows(
                COLLECTION_ISSUES_CSV,
                ISSUE_FIELDS,
                issue_rows,
            )

            if (
                position < len(sources)
                and args.delay > 0
            ):
                time.sleep(args.delay)

    finally:
        session.close()

        if browser is not None:
            browser.close()

    print()
    print("Collection summary")
    print(f"  collected: {collected}")
    print(f"  cached/skipped: {skipped}")
    print(f"  failed: {failed}")
    print(
        f"  manifest: "
        f"{FETCH_MANIFEST_CSV}"
    )
    print(
        f"  discovered links: "
        f"{DISCOVERED_LINKS_CSV}"
    )
    print(
        f"  issues: "
        f"{COLLECTION_ISSUES_CSV}"
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())