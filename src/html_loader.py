from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup, UnicodeDammit
from playwright.sync_api import (
    Browser,
    BrowserContext,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)
from trafilatura import extract

from src.cleaner import normalise_text
from src.collection_config import DEFAULT_PLAYWRIGHT_TIMEOUT_MS
from src.source_catalog import is_allowed_domain

_BLOCK_PATTERNS = (
    "verify that you're not a robot",
    "verify that you are not a robot",
    "checking your browser",
    "access denied",
    "captcha",
    "enable javascript and cookies to continue",
    "javascript is disabled",
    "cf-chl-",
)

_SKIPPED_EXTENSIONS = {
    ".7z",
    ".avi",
    ".css",
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


@dataclass(slots=True)
class BrowserFetchResult:
    html: str
    final_url: str
    http_status: int | None
    content_type: str


@dataclass(slots=True)
class HTMLExtraction:
    markdown: str
    page_title: str
    block_page_suspected: bool
    used_fallback_extractor: bool


def decode_html(data: bytes) -> str:
    detected = UnicodeDammit(data)

    return (
        detected.unicode_markup
        or data.decode("utf-8", errors="replace")
    )


def extract_page_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    if soup.title and soup.title.string:
        return normalise_text(soup.title.string)

    heading = soup.find("h1")

    if heading:
        return normalise_text(
            heading.get_text(" ", strip=True)
        )

    return ""


def _fallback_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    container = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    for tag in container.find_all(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
        ]
    ):
        tag.decompose()

    blocks: list[str] = []

    for element in container.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "p",
            "li",
            "th",
            "td",
        ]
    ):
        value = normalise_text(
            element.get_text(" ", strip=True)
        )

        if not value:
            continue

        if element.name in {"h1", "h2", "h3", "h4"}:
            level = int(element.name[1])
            blocks.append(
                f"{'#' * level} {value}"
            )

        elif element.name == "li":
            blocks.append(f"- {value}")

        else:
            blocks.append(value)

    return normalise_text(
        "\n\n".join(blocks)
    )


def extract_html(
    html: str,
    *,
    url: str,
) -> HTMLExtraction:
    markdown = extract(
        html,
        url=url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        include_links=True,
        include_images=False,
        deduplicate=True,
        favor_precision=True,
    )

    used_fallback = False

    if (
        not markdown
        or len(normalise_text(markdown)) < 200
    ):
        markdown = _fallback_main_text(html)
        used_fallback = True

    cleaned = normalise_text(markdown or "")

    lower_html = html.lower()
    lower_text = cleaned.lower()

    block_signal = any(
        pattern in lower_html
        or pattern in lower_text
        for pattern in _BLOCK_PATTERNS
    )

    block_page_suspected = (
        block_signal
        and len(cleaned) < 1_500
    )

    return HTMLExtraction(
        markdown=cleaned,
        page_title=extract_page_title(html),
        block_page_suspected=block_page_suspected,
        used_fallback_extractor=used_fallback,
    )


def discover_links(
    html: str,
    *,
    base_url: str,
) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")

    container = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    discovered: dict[str, dict[str, str]] = {}

    for anchor in container.find_all(
        "a",
        href=True,
    ):
        raw_href = (
            anchor.get("href")
            or ""
        ).strip()

        if (
            not raw_href
            or raw_href.startswith(
                (
                    "mailto:",
                    "tel:",
                    "javascript:",
                    "#",
                )
            )
        ):
            continue

        absolute = urljoin(
            base_url,
            raw_href,
        )

        clean_url, _fragment = urldefrag(
            absolute
        )

        parsed = urlparse(clean_url)

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
        ):
            continue

        if not is_allowed_domain(parsed.hostname):
            continue

        path_lower = parsed.path.lower()

        extension_match = re.search(
            r"(\.[a-z0-9]{1,6})$",
            path_lower,
        )

        extension = (
            extension_match.group(1)
            if extension_match
            else ""
        )

        if extension in _SKIPPED_EXTENSIONS:
            continue

        probable_type = (
            "pdf"
            if path_lower.endswith(".pdf")
            else "html"
        )

        anchor_text = normalise_text(
            anchor.get_text(
                " ",
                strip=True,
            )
        )[:500]

        discovered[clean_url] = {
            "url": clean_url,
            "anchor_text": anchor_text,
            "domain": parsed.hostname.lower(),
            "probable_type": probable_type,
        }

    return sorted(
        discovered.values(),
        key=lambda item: item["url"],
    )


class BrowserHTMLFetcher:
    def __init__(
        self,
        *,
        headed: bool = False,
        timeout_ms: int = DEFAULT_PLAYWRIGHT_TIMEOUT_MS,
        user_agent: str,
    ) -> None:
        self.headed = headed
        self.timeout_ms = timeout_ms
        self.user_agent = user_agent

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    def start(self) -> None:
        if self._context is not None:
            return

        self._playwright = sync_playwright().start()

        self._browser = (
            self._playwright
            .chromium
            .launch(
                headless=not self.headed
            )
        )

        self._context = self._browser.new_context(
            user_agent=self.user_agent,
            locale="en-GB",
            java_script_enabled=True,
        )

        self._context.set_default_timeout(
            self.timeout_ms
        )

    def fetch(
        self,
        url: str,
    ) -> BrowserFetchResult:
        self.start()

        assert self._context is not None

        page = self._context.new_page()

        try:
            response = page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )

            try:
                page.wait_for_load_state(
                    "networkidle",
                    timeout=10_000,
                )
            except PlaywrightTimeoutError:
                pass

            html = page.content()
            final_url = page.url

            status = (
                response.status
                if response is not None
                else None
            )

            headers = (
                response.headers
                if response is not None
                else {}
            )

            content_type = headers.get(
                "content-type",
                "",
            )

            return BrowserFetchResult(
                html=html,
                final_url=final_url,
                http_status=status,
                content_type=content_type,
            )

        finally:
            page.close()

    def close(self) -> None:
        if self._context is not None:
            self._context.close()

        if self._browser is not None:
            self._browser.close()

        if self._playwright is not None:
            self._playwright.stop()

        self._context = None
        self._browser = None
        self._playwright = None

    def __enter__(
        self,
    ) -> "BrowserHTMLFetcher":
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        self.close()