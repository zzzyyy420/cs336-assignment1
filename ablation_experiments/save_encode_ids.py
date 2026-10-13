"""Prepare token arrays using the Assignment 1 tokenizer and data pipeline."""

import sys

from cs336_basics.train import main


if __name__ == "__main__":
    main(["--prepare-data", *sys.argv[1:]])
