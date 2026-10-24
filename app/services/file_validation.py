"""Pure upload validation: filename hygiene, type allow-list, content sniffing."""

import hashlib
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath

from fastapi import UploadFile

from app.core.config import Settings
from app.core.exceptions import (
    BadRequestError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)

MAX_FILENAME_LENGTH = 255
_READ_CHUNK = 1024 * 1024
# Control (Cc), format (Cf, includes the right-to-left override used in filename
# spoofing), surrogate, private-use and unassigned characters are all removed.
_UNSAFE_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}


@dataclass(frozen=True)
class ValidatedUpload:
    file_type: str
    original_filename: str
    data: bytes
    size: int
    sha256: str


def _strip_unsafe(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if unicodedata.category(ch) not in _UNSAFE_CATEGORIES).strip()


def sanitize_filename(raw: str | None) -> str:
    """Display-only name. It is never used to build a filesystem path."""
    name = _strip_unsafe((raw or "").replace("\\", "/").split("/")[-1])
    if len(name) > MAX_FILENAME_LENGTH:
        suffix = PurePosixPath(name).suffix[:16]
        name = name[: MAX_FILENAME_LENGTH - len(suffix)] + suffix
    return name


def clean_display_name(raw: str | None) -> str:
    return _strip_unsafe(raw or "")[:MAX_FILENAME_LENGTH]


def _check_pdf(data: bytes) -> None:
    # The spec allows junk before the header, but only within the first 1 KB.
    if b"%PDF-" not in data[:1024]:
        raise BadRequestError("The file does not look like a valid PDF.")


def _check_txt(data: bytes) -> None:
    if b"\x00" in data:
        raise BadRequestError("The file is not a plain-text file.")
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise BadRequestError("Text files must be UTF-8 encoded.") from None


# Adding DOCX later: add "docx": _check_docx here (ZIP signature + [Content_Types].xml)
# and a matching extractor in Phase 5. Nothing else in the upload path changes.
CONTENT_CHECKS: dict[str, Callable[[bytes], None]] = {"pdf": _check_pdf, "txt": _check_txt}


def validate_upload(filename: str | None, data: bytes, settings: Settings) -> ValidatedUpload:
    original = sanitize_filename(filename)
    if not original:
        raise BadRequestError("A filename is required.")

    extension = PurePosixPath(original).suffix.removeprefix(".").lower()
    allowed = {t.lower() for t in settings.allowed_file_types} & CONTENT_CHECKS.keys()
    if extension not in allowed:
        raise UnsupportedMediaTypeError(
            f"Unsupported file type. Allowed: {', '.join(sorted(allowed))}."
        )
    if not data:
        raise BadRequestError("The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"The uploaded file exceeds the {settings.max_upload_mb} MB limit."
        )

    CONTENT_CHECKS[extension](data)
    return ValidatedUpload(
        file_type=extension,
        original_filename=original,
        data=data,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


async def read_upload(upload: UploadFile, settings: Settings) -> bytes:
    """Read in chunks and stop as soon as the limit is exceeded."""
    buffer = bytearray()
    while chunk := await upload.read(_READ_CHUNK):
        buffer.extend(chunk)
        if len(buffer) > settings.max_upload_bytes:
            raise PayloadTooLargeError(
                f"The uploaded file exceeds the {settings.max_upload_mb} MB limit."
            )
    return bytes(buffer)
