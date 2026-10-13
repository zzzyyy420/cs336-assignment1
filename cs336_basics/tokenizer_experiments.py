"""Reproducible CPU measurements for the assignment's tokenizer experiments."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


def iter_documents(path, chunk_size=1024 * 1024):
    marker = "<|endoftext|>"
    pending = ""
    with Path(path).open(encoding="utf-8", newline="") as source:
        while chunk := source.read(chunk_size):
            pending += chunk
            pieces = pending.split(marker)
            yield from pieces[:-1]
            pending = pieces[-1]
        if pending:
            yield pending


def sample_documents(path, count=10, seed=42):
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    samples = []
    seen = 0
    for index, text in enumerate(iter_documents(path)):
        if not text.strip():
            continue
        seen += 1
        sample = {"document_index": index, "text": text}
        if len(samples) < count:
            samples.append(sample)
        else:
            slot = rng.randrange(seen)
            if slot < count:
                samples[slot] = sample
    if seen < count:
        raise ValueError(f"Requested {count} documents, but corpus contains only {seen}")
    return sorted(samples, key=lambda sample: sample["document_index"])


def measure_compression(tokenizer, samples):
    per_document = []
    for sample in samples:
        text = sample["text"]
        if not text:
            continue
        ids = tokenizer.encode(text)
        if tokenizer.decode(ids) != text:
            raise ValueError("Tokenizer roundtrip failed for a sampled document")
        byte_count = len(text.encode("utf-8"))
        per_document.append({
            "document_index": sample["document_index"],
            "utf8_bytes": byte_count,
            "tokens": len(ids),
            "bytes_per_token": byte_count / len(ids),
        })
    if not per_document:
        raise ValueError("Provide at least one non-empty document")
    byte_count = sum(row["utf8_bytes"] for row in per_document)
    token_count = sum(row["tokens"] for row in per_document)
    return {
        "documents": len(per_document), "utf8_bytes": byte_count, "tokens": token_count,
        "bytes_per_token": byte_count / token_count, "per_document": per_document,
    }


def measure_throughput(tokenizer, samples, repeats=5):
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    texts = [sample["text"] for sample in samples if sample["text"]]
    byte_count = sum(len(text.encode("utf-8")) for text in texts)
    if not byte_count:
        raise ValueError("Provide at least one non-empty document")
    expected = [tokenizer.encode(text) for text in texts]
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        actual = [tokenizer.encode(text) for text in texts]
        timings.append(time.perf_counter() - start)
        if actual != expected:
            raise ValueError("Tokenizer outputs changed between repetitions")
    seconds = statistics.median(timings)
    throughput = byte_count / seconds
    return {
        "repeats": repeats, "seconds_per_repeat": timings, "median_seconds": seconds,
        "utf8_bytes_per_repeat": byte_count,
        "tokens_per_repeat": sum(len(ids) for ids in expected),
        "bytes_per_second": throughput,
        "tokens_per_second": sum(len(ids) for ids in expected) / seconds,
        "pile_825_decimal_gb_seconds": 825 * 10**9 / throughput,
        "scope": "warm tokenizer, sampled documents; excludes loading and disk writes",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path("results/tokenizer_compression/samples.json"))
    parser.add_argument("--output", type=Path, default=Path("results/tokenizer_compression/completed_results.json"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    results = []
    longest = {}
    for name, prefix in (("TinyStories", "tinystories"), ("OWT", "owt")):
        tokenizer = Tokenizer.from_files(f"{prefix}_vocab.pkl", f"{prefix}_merges.pkl", ["<|endoftext|>"])
        normal_tokens = [value for value in tokenizer.vocab.values() if value != b"<|endoftext|>"]
        max_length = max(map(len, normal_tokens))
        longest[name] = {
            "vocab_size": len(tokenizer.vocab), "merge_count": len(tokenizer.merges),
            "longest_byte_length": max_length,
            "tokens": [repr(value) for value in normal_tokens if len(value) == max_length],
        }
        for dataset, documents in samples["documents"].items():
            measurement = {"dataset": dataset, "tokenizer": name, "vocab_size": len(tokenizer.vocab)}
            measurement.update(measure_compression(tokenizer, documents))
            measurement["throughput"] = measure_throughput(tokenizer, documents, args.repeats)
            results.append(measurement)
            print(f"{name} on {dataset}: {measurement['bytes_per_token']:.4f} bytes/token, "
                  f"{measurement['throughput']['bytes_per_second'] / 1e6:.3f} MB/s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "sample_file": str(args.samples), "seed": samples["seed"],
        "formula": "total UTF-8 bytes / total token IDs; document delimiters excluded",
        "longest_tokens": longest, "results": results,
    }, indent=2), encoding="utf-8")
    print(f"Saved measurements to {args.output}")


if __name__ == "__main__":
    main()
