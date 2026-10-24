import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.core.exceptions import BadRequestError, PayloadTooLargeError, UnsupportedMediaTypeError
from app.core.upload_limit import UploadSizeLimitMiddleware
from app.services.file_validation import sanitize_filename, validate_upload

settings = get_settings()


def test_sanitize_strips_path_components():
    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"


def test_sanitize_strips_windows_paths():
    assert sanitize_filename("C:\\Users\\me\\report.pdf") == "report.pdf"


def test_sanitize_removes_control_and_bidi_characters():
    assert sanitize_filename("inv\u202eoice\x00.txt") == "invoice.txt"


def test_sanitize_truncates_long_names_keeping_extension():
    result = sanitize_filename("a" * 400 + ".pdf")

    assert len(result) == 255
    assert result.endswith(".pdf")


def test_validate_accepts_txt_with_bom():
    data = b"\xef\xbb\xbfhello"

    result = validate_upload("notes.txt", data, settings)

    assert result.file_type == "txt"
    assert result.size == len(data)


def test_validate_accepts_pdf_header_within_first_kb():
    data = b"\n" * 100 + b"%PDF-1.7\n"

    assert validate_upload("a.pdf", data, settings).file_type == "pdf"


def test_validate_extension_is_case_insensitive():
    assert validate_upload("REPORT.TXT", b"hi", settings).file_type == "txt"


@pytest.mark.parametrize("filename", ["a.docx", "a.exe", "noextension"])
def test_validate_rejects_unlisted_extension(filename):
    with pytest.raises(UnsupportedMediaTypeError):
        validate_upload(filename, b"hello", settings)


def test_validate_rejects_pdf_without_header():
    with pytest.raises(BadRequestError):
        validate_upload("fake.pdf", b"this is just text", settings)


def test_validate_rejects_binary_data_named_txt():
    with pytest.raises(BadRequestError):
        validate_upload("a.txt", b"abc\x00def", settings)


def test_validate_rejects_non_utf8_text():
    with pytest.raises(BadRequestError):
        validate_upload("a.txt", "café".encode("latin-1"), settings)


def test_validate_rejects_empty_file():
    with pytest.raises(BadRequestError):
        validate_upload("a.txt", b"", settings)


def test_validate_rejects_oversized_file():
    small = settings.model_copy(update={"max_upload_mb": 1})

    with pytest.raises(PayloadTooLargeError):
        validate_upload("a.txt", b"a" * (1024 * 1024 + 1), small)


async def test_content_length_over_limit_is_rejected_early():
    app = FastAPI()

    @app.post("/up")
    async def up() -> dict[str, bool]:
        return {"ok": True}

    app.add_middleware(UploadSizeLimitMiddleware, path="/up", max_bytes=100)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        too_big = await client.post("/up", content=b"x" * 101)
        fine = await client.post("/up", content=b"x" * 50)

    assert too_big.status_code == 413
    assert too_big.json()["error"]["code"] == "payload_too_large"
    assert fine.status_code == 200
