import uuid

import pytest
from pydantic import ValidationError

from app.schemas.rag import AskRequest


def test_whitespace_is_stripped():
    assert AskRequest(question="  hello there \n").question == "hello there"


@pytest.mark.parametrize("value", ["", "   ", "\x00\x07"])
def test_blank_questions_are_rejected(value):
    with pytest.raises(ValidationError):
        AskRequest(question=value)


def test_overlong_questions_are_rejected():
    with pytest.raises(ValidationError):
        AskRequest(question="x" * 2001)


def test_control_characters_are_removed():
    assert AskRequest(question="hello\x00 wor\x07ld").question == "hello world"


def test_newlines_are_kept():
    assert AskRequest(question="line one\nline two").question == "line one\nline two"


def test_zero_width_joiner_needed_by_indic_scripts_is_kept():
    word = "\u0915\u094d\u200d\u0937"

    assert AskRequest(question=word).question == word


def test_unknown_fields_such_as_user_id_are_rejected():
    with pytest.raises(ValidationError):
        AskRequest(question="hi", user_id="someone-else")


def test_document_id_must_be_a_uuid():
    with pytest.raises(ValidationError):
        AskRequest(question="hi", document_id="not-a-uuid")


def test_document_id_is_optional():
    request = AskRequest(question="hi")

    assert request.document_id is None
    assert AskRequest(question="hi", document_id=str(uuid.uuid4())).document_id is not None
