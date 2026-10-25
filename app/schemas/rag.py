import unicodedata
import uuid
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

MAX_QUESTION_CHARS = 2000
Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
]

# Control characters (except newline and tab), surrogates, private-use and unassigned
# code points are removed. Format characters (Cf) are kept: Indic scripts need the
# zero-width joiner.
_DROP_CATEGORIES = {"Cc", "Cs", "Co", "Cn"}


class AskRequest(BaseModel):
    # forbid: a client that sends "user_id" (or anything unknown) gets a 422.
    model_config = ConfigDict(extra="forbid")

    question: Question = Field(
        description="The question to answer from your documents",
        examples=["What are the main topics covered in this document?"],
    )
    document_id: uuid.UUID | None = Field(
        default=None,
        description="Search only this document. Omit to search all of your completed documents.",
    )

    @field_validator("question")
    @classmethod
    def _clean_question(cls, value: str) -> str:
        cleaned = "".join(
            ch for ch in value if ch in "\n\t" or unicodedata.category(ch) not in _DROP_CATEGORIES
        ).strip()
        if not cleaned:
            raise ValueError("Question must not be blank.")
        return cleaned


class SourceRead(BaseModel):
    ref: int  # matches the [n] citations in the answer
    document_id: uuid.UUID
    document_name: str
    chunk_index: int
    content: str
    score: float  # cosine similarity reported by Qdrant


class AskResponse(BaseModel):
    question: str
    answer: str
    context_found: bool  # False = nothing relevant was found; sources is empty
    sources: list[SourceRead]
