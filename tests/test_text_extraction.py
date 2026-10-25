import pytest

from app.core.config import get_settings
from app.services.errors import PermanentProcessingError
from app.services.text_extraction import extract_text, normalize_text
from tests.helpers import make_pdf

settings = get_settings()


def test_normalize_removes_control_and_bidi_characters():
    assert normalize_text("a\x00b\u202ec\u00ad d\x0bx") == "abc dx"


def test_normalize_unifies_newlines_and_collapses_blank_lines():
    raw = "one\r\ntwo\r\n\r\n\r\n\r\nthree  \n  four\tfive"

    assert normalize_text(raw) == "one\ntwo\n\nthree\nfour five"


def test_normalize_keeps_zero_width_joiner_needed_by_indic_scripts():
    word = "\u0915\u094d\u200d\u0937"

    assert normalize_text(word) == word


def test_normalize_composes_unicode():
    assert normalize_text("e\u0301") == "\u00e9"


async def test_txt_extraction_strips_bom_and_normalizes():
    assert await extract_text("txt", b"\xef\xbb\xbfHello  world\r\n", settings) == "Hello world"


async def test_blank_text_is_a_permanent_error():
    with pytest.raises(PermanentProcessingError, match="No readable text"):
        await extract_text("txt", b" \n\t ", settings)


async def test_pdf_text_is_extracted():
    text = await extract_text("pdf", make_pdf("Hello PDF world"), settings)

    assert "Hello PDF world" in text


async def test_corrupt_pdf_is_a_permanent_error():
    with pytest.raises(PermanentProcessingError):
        await extract_text("pdf", b"%PDF-1.4 this is not really a pdf", settings)


async def test_pdf_with_too_many_pages_is_rejected():
    limited = settings.model_copy(update={"max_pdf_pages": 0})

    with pytest.raises(PermanentProcessingError, match="too many pages"):
        await extract_text("pdf", make_pdf("Hello"), limited)


async def test_pdf_with_too_much_text_is_rejected():
    limited = settings.model_copy(update={"max_extracted_chars": 5})

    with pytest.raises(PermanentProcessingError, match="too much text"):
        await extract_text("pdf", make_pdf("Hello PDF world"), limited)


async def test_unsupported_type_is_a_permanent_error():
    with pytest.raises(PermanentProcessingError):
        await extract_text("docx", b"x", settings)
