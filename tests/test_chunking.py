import pytest

from app.services.chunking import split_text

WORDS = " ".join(f"w{i:04d}" for i in range(400))  # 400 distinct 5-char words


def test_empty_or_blank_text_has_no_chunks():
    assert split_text("", 100, 10) == []
    assert split_text("  \n  ", 100, 10) == []


def test_short_text_is_a_single_chunk():
    chunks = split_text("hello world", 100, 10)

    assert [c.text for c in chunks] == ["hello world"]
    assert chunks[0].index == 0


def test_chunks_never_exceed_the_size_limit():
    chunks = split_text(WORDS, 100, 20)

    assert len(chunks) > 10
    assert all(len(c.text) <= 100 for c in chunks)


def test_indexes_are_sequential_from_zero():
    chunks = split_text(WORDS, 100, 20)

    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_prefers_paragraph_boundaries():
    text = "a" * 60 + "\n\n" + "b" * 60

    chunks = split_text(text, 100, 10)

    assert [c.text for c in chunks] == ["a" * 60, "b" * 60]


def test_consecutive_chunks_overlap():
    chunks = split_text(WORDS, 100, 20)

    for current, following in zip(chunks, chunks[1:], strict=False):
        assert current.text.split()[-1] in following.text.split()


def test_no_words_are_lost():
    chunks = split_text(WORDS, 100, 20)

    seen = {word for c in chunks for word in c.text.split()}
    assert seen == set(WORDS.split())


def test_text_without_whitespace_still_terminates():
    chunks = split_text("x" * 1000, 100, 10)

    assert len(chunks) >= 10
    assert all(len(c.text) <= 100 for c in chunks)
    assert chunks[-1].text.endswith("x")


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        split_text("some text", size, overlap)


def test_breaks_on_devanagari_sentence_endings():
    text = "यह एक वाक्य है। " * 30

    chunks = split_text(text, 100, 10)

    assert len(chunks) > 1
    assert all(c.text.endswith("।") for c in chunks)
