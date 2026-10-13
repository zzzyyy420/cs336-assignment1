# Ablation experiments

These files adapt the earlier `chapter1/hw7` scripts to the Assignment 1
components in `cs336_basics`. `transformermodule.py` keeps pre-RMSNorm blocks;
`transformermodule_withoutrmsnorm.py` removes RMSNorm in the blocks and at the
output. Both use the local attention and block variants in this folder. The
training and generation entry points reuse the `cs336_basics` pipelines
and their matching baseline and `--no-rmsnorm` settings.

Run from the project root. Prepare tokens once if they do not exist:

```sh
uv run python -m ablation_experiments.save_encode_ids \
  --train-data data/TinyStoriesV2-GPT4-train.txt \
  --valid-data data/TinyStoriesV2-GPT4-valid.txt
```

Run the baseline and no-RMSNorm experiment with the same settings and separate
output directories:

```sh
uv run python -m ablation_experiments.train_script \
  --device cuda:0 --train-steps 2000 --batch-size 4
```

`train_script.py` starts both full runs when invoked and refuses to append to
an existing result directory. Use `--output-root` for a new run. For one run,
invoke `uv run python -m ablation_experiments.final_train` directly and pass
`--no-rmsnorm` if needed. The default token files are
`data/tinystories_train.bin` and `data/tinystories_valid.bin`.

Generate from a completed run with its matching config and tokenizer:

```sh
uv run python -m ablation_experiments.final_inference \
  --checkpoint checkpoints/ablations/no_rmsnorm/final.pt \
  --vocab tinystories_vocab.pkl --merges tinystories_merges.pkl
```

Compare validation losses only at matched token budgets and learning rates.
