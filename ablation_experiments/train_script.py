"""Run matched baseline and no-RMSNorm experiments in separate directories."""

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run baseline and no-RMSNorm ablations")
    parser.add_argument("--output-root", type=Path, default=Path("checkpoints/ablations"))
    args, training_args = parser.parse_known_args(argv)
    if any(arg.startswith("--output-dir") or arg.startswith("--no-rmsnorm") for arg in training_args):
        parser.error("output-dir and no-rmsnorm are managed by train_script")

    for name in ("baseline", "no_rmsnorm"):
        output_dir = args.output_root / name
        if output_dir.exists() and any(output_dir.iterdir()):
            parser.error(f"{output_dir} already contains results; choose a new --output-root")

    for name, variant_args in (("baseline", []), ("no_rmsnorm", ["--no-rmsnorm"])):
        command = [
            sys.executable, "-m", "ablation_experiments.final_train",
            "--output-dir", str(args.output_root / name), *training_args, *variant_args,
        ]
        print(f"Running {name}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
