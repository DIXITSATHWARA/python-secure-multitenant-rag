"""Split text into overlapping chunks, preferring natural boundaries."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


# Best break first: paragraph, line, sentence end (Latin, Devanagari danda, CJK), then space.
_TIERS: tuple[tuple[str, ...], ...] = (
    ("\n\n",),
    ("\n",),
    (". ", "? ", "! ", "। ", "。", "？", "！"),
    (" ",),
)
_WHITESPACE = re.compile(r"\s")


def _find_break(text: str, hard_end: int, min_pos: int) -> int:
    """Latest good break point in [min_pos, hard_end), or hard_end if there is none."""
    for separators in _TIERS:
        best = -1
        for sep in separators:
            idx = text.rfind(sep, min_pos, hard_end)
            if idx != -1:
                best = max(best, idx + len(sep))
        if best != -1:
            return best
    return hard_end


def split_text(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= overlap < chunk_size")

    length = len(text)
    chunks: list[Chunk] = []
    start = 0
    while start < length:
        hard_end = min(start + chunk_size, length)
        # Never break in the first half of the window, which would create tiny chunks.
        end = (
            hard_end if hard_end == length else _find_break(text, hard_end, start + chunk_size // 2)
        )

        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(index=len(chunks), text=piece))
        if end >= length:
            break

        next_start = max(end - overlap, start + 1)  # always make progress
        # Start the next chunk at a word boundary rather than mid-word.
        if not text[next_start - 1].isspace():
            match = _WHITESPACE.search(text, next_start, end)
            if match:
                next_start = match.end()
        start = next_start
    return chunks
