# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Test adapter functions are in [./tests/adapters.py](./tests/adapters.py).

## Local implementation

The tokenizer, Transformer components, optimizer, training loop, and generation
entry point are in [cs336_basics](./cs336_basics). Run commands from this project
root. To keep the virtual environment outside the checkout, set
`UV_PROJECT_ENVIRONMENT` to your chosen environment path before `uv run`.

```sh
uv run pytest tests -q
uv run python -m cs336_basics.train --help
uv run python -m cs336_basics.generate --help
uv run python -m scripts.run_experiments
```

The last command writes a plan only; training starts only with `--run`. The
[ablation experiments](./ablation_experiments/README.md) have separate model
variants and command entry points. Local datasets, token arrays, vocabularies,
checkpoints, virtual environments, and generated results are intentionally
excluded from Git. Download or prepare the data and train a model before using
the generation command.

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
