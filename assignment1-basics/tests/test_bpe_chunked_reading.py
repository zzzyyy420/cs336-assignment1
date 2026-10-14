import pytest
import regex

from cs336_basics.bpe import iter_training_documents, run_train_bpe


@pytest.mark.parametrize("read_chunk_size", [1, 2, 7, 1024])
@pytest.mark.parametrize(
    ("text", "special_tokens"),
    [
        ("hello<|endoftext|>\u725b\U0001f600 world<|endoftext|>\n", ["<|endoftext|>"]),
        ("<|endoftext|><|endoftext|>tail", ["<|endoftext|>"]),
        ("first<tag>longsecond<tag>third", ["<tag>", "<tag>long"]),
        ("unbroken \u725b\U0001f600 text\n\n", []),
        ("no document separators here", ["<|endoftext|>"]),
        ("", ["<|endoftext|>"]),
    ],
)
def test_chunked_documents_match_whole_text(tmp_path, text, special_tokens, read_chunk_size):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text(text, encoding="utf-8")
    if special_tokens:
        pattern = "|".join(map(regex.escape, sorted(special_tokens, key=len, reverse=True)))
        expected = regex.split(pattern, text)
    else:
        expected = [text]

    actual = list(iter_training_documents(input_path, special_tokens, read_chunk_size))
    assert [document for document in actual if document] == [document for document in expected if document]


def test_training_is_independent_of_read_chunk_size(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "corpus.txt"
    input_path.write_text(
        "hello hello<|endoftext|>world \u725b\U0001f600\n\n<|endoftext|>hello world",
        encoding="utf-8",
    )
    expected = run_train_bpe(input_path, 280, ["<|endoftext|>"], read_chunk_size=1024)
    for read_chunk_size in (1, 2, 7):
        assert run_train_bpe(input_path, 280, ["<|endoftext|>"], read_chunk_size=read_chunk_size) == expected


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_training_documents(tmp_path / "missing.txt", ["<|endoftext|>"]))


def test_empty_special_token_raises(tmp_path):
    with pytest.raises(ValueError, match="empty strings"):
        list(iter_training_documents(tmp_path / "unused.txt", [""]))
