import pytest

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.tokenizer_experiments import measure_compression, sample_documents


def test_sampling_is_reproducible_and_keeps_complete_documents(tmp_path):
    documents = [f" story {index}\n" for index in range(30)]
    path = tmp_path / "corpus.txt"
    path.write_text("<|endoftext|>".join(documents), encoding="utf-8")
    samples = sample_documents(path, 10, 42)
    assert samples == sample_documents(path, 10, 42)
    assert len(samples) == 10
    assert len({sample["document_index"] for sample in samples}) == 10
    for sample in samples:
        assert sample["text"] == documents[sample["document_index"]]


def test_sampling_rejects_too_few_documents(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("one<|endoftext|>two", encoding="utf-8")
    with pytest.raises(ValueError, match="only 2"):
        sample_documents(path, 10, 42)


def test_compression_counts_utf8_bytes_and_uses_total_ratio():
    vocab = {index: bytes([index]) for index in range(256)}
    vocab[256] = b"\xe7\x89"
    vocab[257] = b"\xe7\x89\x9b"
    tokenizer = Tokenizer(vocab, [(b"\xe7", b"\x89"), (b"\xe7\x89", b"\x9b")])
    samples = [{"document_index": 0, "text": "a"}, {"document_index": 1, "text": "\u725b\u725b"}]
    result = measure_compression(tokenizer, samples)
    assert result["utf8_bytes"] == 7
    assert result["tokens"] == 3
    assert result["bytes_per_token"] == pytest.approx(7 / 3)
    assert result["per_document"][1]["utf8_bytes"] == 6


def test_compression_rejects_empty_samples():
    with pytest.raises(ValueError, match="non-empty document"):
        measure_compression(Tokenizer({}, []), [])
