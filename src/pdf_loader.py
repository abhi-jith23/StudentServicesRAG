from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.cleaner import normalise_text
from src.collection_config import (
    PDF_MIN_TEXT_LENGTH,
    PDF_SHORT_PAGE_THRESHOLD,
)


@dataclass(slots=True)
class PDFExtraction:
    markdown: str
    page_count: int
    text_length: int
    extraction_status: str
    issues: list[str]


def extract_pdf(
    path: Path | str,
) -> PDFExtraction:
    path = Path(path)
    issues: list[str] = []

    try:
        reader = PdfReader(
            str(path),
            strict=False,
        )
    except PdfReadError as exc:
        raise ValueError(
            f"Could not read PDF {path}: {exc}"
        ) from exc

    if reader.is_encrypted:
        try:
            result = reader.decrypt("")

            if result == 0:
                raise ValueError(
                    "The PDF is encrypted and "
                    "could not be opened."
                )

            issues.append(
                "PDF was encrypted but opened "
                "with an empty password."
            )

        except Exception as exc:
            raise ValueError(
                "The PDF is encrypted and "
                f"cannot be read: {exc}"
            ) from exc

    pages: list[str] = []
    short_pages = 0
    extracted_character_count = 0

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = ""

        try:
            text = page.extract_text(
                extraction_mode="layout",
                layout_mode_space_vertically=False,
            ) or ""

        except Exception as layout_exc:
            issues.append(
                f"Page {page_number}: layout "
                "extraction failed; plain extraction "
                f"used ({layout_exc})."
            )

            try:
                text = page.extract_text() or ""

            except Exception as plain_exc:
                issues.append(
                    f"Page {page_number}: text "
                    f"extraction failed ({plain_exc})."
                )
                text = ""

        cleaned = normalise_text(text)

        if len(cleaned) < PDF_SHORT_PAGE_THRESHOLD:
            short_pages += 1

            issues.append(
                f"Page {page_number}: fewer than "
                f"{PDF_SHORT_PAGE_THRESHOLD} "
                "extractable characters."
            )

        extracted_character_count += len(cleaned)

        page_body = (
            cleaned
            or "[No extractable text on this page.]"
        )

        pages.append(
            f"## Page {page_number}\n\n"
            f"{page_body}"
        )

    page_count = len(reader.pages)

    markdown = normalise_text(
        "\n\n".join(pages)
    )

    text_length = extracted_character_count

    if page_count == 0:
        status = "needs_manual_review"

        issues.append(
            "The PDF contains zero pages."
        )

    elif text_length < PDF_MIN_TEXT_LENGTH:
        status = "scanned_pdf_suspected"

        issues.append(
            f"Only {text_length} extractable "
            "characters were found in the whole PDF."
        )

    elif short_pages / page_count >= 0.60:
        status = "scanned_pdf_suspected"

        issues.append(
            "At least 60% of pages contain "
            "very little extractable text."
        )

    elif issues:
        status = "needs_manual_review"

    else:
        status = "extraction_ok"

    return PDFExtraction(
        markdown=markdown,
        page_count=page_count,
        text_length=text_length,
        extraction_status=status,
        issues=issues,
    )