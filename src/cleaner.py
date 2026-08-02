from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalise_text(text: str) -> str:
    text = (
        (text or "")
        .replace("\u00a0", " ")
        .replace("\u200b", "")
    )

    text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    text = "\n".join(
        line.rstrip()
        for line in text.splitlines()
    )

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def build_markdown_document(
    metadata: dict[str, Any],
    body: str,
    *,
    prepend_title: bool = True,
) -> str:
    clean_body = normalise_text(body)
    title = str(metadata.get("title") or "").strip()

    if prepend_title and title:
        first_nonempty = next(
            (
                line.strip()
                for line in clean_body.splitlines()
                if line.strip()
            ),
            "",
        )

        if not first_nonempty.startswith("# "):
            if clean_body:
                clean_body = f"# {title}\n\n{clean_body}"
            else:
                clean_body = f"# {title}"

    front_matter = yaml.safe_dump(
        metadata,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()

    return (
        f"---\n"
        f"{front_matter}\n"
        f"---\n\n"
        f"{clean_body}\n"
    )


def atomic_write_bytes(
    path: Path,
    data: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(data)
        temp_name = handle.name

    os.replace(temp_name, path)


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    atomic_write_bytes(
        path,
        text.encode("utf-8"),
    )