from __future__ import annotations

import os


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    from cs336_basics.extract_text import run_extract_text_from_html_bytes as extract_text

    return extract_text(html_bytes)


def run_identify_language(text: str) -> tuple[str, float]:
    from cs336_basics.language_identification import run_identify_language as identify_language

    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    from cs336_basics.filter_email import run_mask_emails as mask_emails

    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    from cs336_basics.filter_phone_number import run_mask_phone_numbers as mask_phone_numbers

    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    from cs336_basics.filter_ip import run_mask_ips as mask_ips

    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[str, float]:
    from cs336_basics.detect_nsfw import run_classify_nsfw as classify_nsfw

    return classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[str, float]:
    from cs336_basics.detect_toxic_speech import run_classify_toxic_speech as classify_toxic_speech

    return classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[str, float]:
    from cs336_basics.quality_classify.classify_quality import run_classify_quality as classify_quality

    return classify_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    from cs336_basics.quality_rule import run_gopher_quality_filter as gopher_quality_filter

    return gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
) -> None:
    from cs336_basics.exact_deduplication import run_exact_line_deduplication as exact_line_deduplication

    exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    from cs336_basics.minhash_deduplication.minhash_deduplication import (
        run_minhash_deduplication as minhash_deduplication,
    )

    minhash_deduplication(
        input_files=input_files,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        jaccard_threshold=jaccard_threshold,
        output_directory=output_directory,
    )
