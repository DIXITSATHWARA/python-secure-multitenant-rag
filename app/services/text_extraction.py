"""Turn stored bytes into clean text. Adding DOCX later means adding one extractor here."""

import asyncio
import io
import logging
import re
import unicodedata
from collections.abc import Callable

from pypdf import PdfReader

from app.core.config import Settings
from app.services.errors import PermanentProcessingError

logger = logging.getLogger(__name__)

NO_TEXT = (
    "No readable text was found in this document. Scanned or image-only files are not supported."
)

# Control characters, soft hyphen, BOM, and bidi overrides. Zero-width joiners (U+200C/D)
# are deliberately kept: scripts like Devanagari need them to render correctly.
_CONTROL_RE = re.compile(
    "[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f\u00ad\ufeff\u200e\u200f\u202a-\u202e\u2066-\u2069]"
)
_SPACES_RE = re.compile("[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_BLANK_LINES_RE = re.compile("\n{3,}")


def normalize_text(text: str) -> str:
    # Lone surrogates (seen in broken PDFs) would break JSON encoding downstream.
    text = text.encode("utf-8", "replace").decode("utf-8")
    text = unicodedata.normalize("NFC", text)
    text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n\n")
        .replace("\x0c", "\n\n")  # form feed = page break in many PDFs
    )
    text = _CONTROL_RE.sub("", text)
    text = _SPACES_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def _extract_txt(data: bytes, settings: Settings) -> str:
    return data.decode("utf-8-sig", errors="replace")


def _extract_pdf(data: bytes, settings: Settings) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PermanentProcessingError("Password-protected PDFs are not supported.")
        if len(reader.pages) > settings.max_pdf_pages:
            raise PermanentProcessingError(
                f"This PDF has too many pages (the limit is {settings.max_pdf_pages})."
            )
        parts: list[str] = []
        total = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            total += len(page_text)
            if total > settings.max_extracted_chars:
                raise PermanentProcessingError("The document contains too much text to index.")
            parts.append(page_text)
    except PermanentProcessingError:
        raise
    except Exception:  # pypdf raises many different exception types on malformed files
        logger.warning("PDF parsing failed", exc_info=True)
        raise PermanentProcessingError("The PDF could not be read. It may be corrupted.") from None
    return "\n\n".join(parts)


EXTRACTORS: dict[str, Callable[[bytes, Settings], str]] = {
    "txt": _extract_txt,
    "pdf": _extract_pdf,
}


async def extract_text(file_type: str, data: bytes, settings: Settings) -> str:
    extractor = EXTRACTORS.get(file_type)
    if extractor is None:
        raise PermanentProcessingError("This file type cannot be processed.")
    # Parsing is CPU-bound, so keep it off the event loop.
    raw = await asyncio.to_thread(extractor, data, settings)
    text = normalize_text(raw)
    if not text:
        raise PermanentProcessingError(NO_TEXT)
    return text
