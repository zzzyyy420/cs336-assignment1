import pickle

import pytest

from cs336_basics.tokenizer import Tokenizer


@pytest.mark.parametrize(
    ("vocab", "special_tokens", "expected"),
    [
        ({0: b"a"}, None, {0: b"a"}),
        ({0: b"a", 5: b"<end>"}, ["<end>"], {0: b"a", 5: b"<end>"}),
        ({0: b"a"}, ["<end>", "<end>"], {0: b"a", 1: b"<end>"}),
        ({0: b"a", 10: b"b"}, ["<end>"], {0: b"a", 10: b"b", 11: b"<end>"}),
    ],
)
def test_special_token_initialization(vocab, special_tokens, expected):
    original = dict(vocab)
    tokenizer = Tokenizer(vocab, [], special_tokens)
    assert tokenizer.vocab == expected
    assert tokenizer.bytes_to_id == {token: token_id for token_id, token in expected.items()}
    assert vocab == original


def test_new_overlapping_special_token_encodes_atomically():
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    vocab[256] = b"<|endoftext|>"
    long_token = "<|endoftext|><|endoftext|>"
    tokenizer = Tokenizer(vocab, [], ["<|endoftext|>", long_token])
    text = "hello" + long_token + "world<|endoftext|>"
    ids = tokenizer.encode(text)
    assert ids.count(257) == 1
    assert ids.count(256) == 1
    assert tokenizer.decode(ids) == text
    assert 257 not in vocab


def test_empty_special_token_raises():
    with pytest.raises(ValueError, match="non-empty strings"):
        Tokenizer({0: b"a"}, [], [""])


def test_from_files_loads_merges_and_special_tokens(tmp_path):
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    vocab[256] = b"ab"
    merges = [(b"a", b"b")]
    vocab_path = tmp_path / "vocab.pkl"
    merges_path = tmp_path / "merges.pkl"
    vocab_path.write_bytes(pickle.dumps(vocab))
    merges_path.write_bytes(pickle.dumps(merges))

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<end>"])
    text = "ab\u725b<end>ab"
    expected_ids = [256, *"\u725b".encode("utf-8"), 257, 256]
    assert tokenizer.merges == merges
    assert tokenizer.encode(text) == expected_ids
    assert tokenizer.decode(expected_ids) == text


def test_from_files_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Tokenizer.from_files(tmp_path / "missing.pkl", tmp_path / "merges.pkl")
