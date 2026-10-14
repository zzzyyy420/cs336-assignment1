"""Write an experiment plan; execute locally only when --run is supplied."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def make_plan(args):
    baseline = args.learning_rate
    definitions = {
        "baseline": [("baseline", 64, baseline, [])],
        "lr": [(f"lr_{lr:g}", 64, lr, []) for lr in (3e-4, 1e-3, 3e-3, 1e-2, 1e-1)],
        "batch": [(f"batch_{batch}", batch, baseline, []) for batch in (1, 4, 16, 32, 64, 128)],
        "ablations": [
            ("no_rmsnorm", 64, baseline, ["--no-rmsnorm"]),
            ("no_rmsnorm_lower_lr", 64, baseline / 10, ["--no-rmsnorm"]),
            ("postnorm", 64, baseline, ["--norm-position", "post"]),
            ("nope", 64, baseline, ["--no-rope"]),
            ("silu", 64, baseline, ["--ffn-type", "silu"]),
        ],
        "owt": [("owt", 64, baseline, [
            "--vocab-size", "32000", "--train-tokens", "data/owt_train.bin",
            "--valid-tokens", "data/owt_valid.bin",
        ])],
        "modification": [("tied_embeddings", 64, baseline, ["--tie-embeddings"])],
    }
    selected = list(definitions) if args.group == "all" else [args.group]
    runs = []
    for group in selected:
        for name, batch, lr, options in definitions[group]:
            steps = max(2, args.token_budget // (batch * 256))
            output = args.output_dir / name
            command = [
                sys.executable, "-m", "cs336_basics.train", "--device", args.device,
                "--precision", args.precision, "--matmul-precision", "high",
                "--batch-size", str(batch), "--train-steps", str(steps),
                "--warmup-steps", str(max(1, steps // 20)),
                "--max-learning-rate", str(lr), "--min-learning-rate", str(lr / 10),
                "--log-interval", str(max(1, steps // 200)),
                "--val-interval", str(max(1, steps // 40)),
                "--eval-batch-size", "4", "--eval-batches", "32",
                "--checkpoint-interval", str(steps + 1),
                "--output-dir", str(output), "--seed", "42", *options,
            ]
            runs.append({
                "name": name, "group": group, "command": command,
                "output_dir": str(output), "steps": steps,
                "tokens": steps * batch * 256, "status": "planned",
            })
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["all", "baseline", "lr", "batch", "ablations", "owt", "modification"], default="all")
    parser.add_argument("--token-budget", type=int, default=327680000)
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Replace with the selected LR after the sweep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/assignment_experiments"))
    parser.add_argument("--manifest", type=Path, default=Path("results/experiment_plan.json"))
    parser.add_argument("--run", action="store_true", help="Launch the planned runs on this machine")
    args = parser.parse_args()
    if args.token_budget < 512 or args.learning_rate <= 0:
        parser.error("token-budget must be at least 512 and learning-rate must be positive")
    runs = make_plan(args)
    manifest = {
        "mode": "execute" if args.run else "plan_only",
        "comparison": "equal token budget, seed 42, fixed sampled validation set (32768 tokens)",
        "learning_rate_note": "1e-3 is a candidate, not a measured optimum; select it from the LR sweep",
        "runs": runs,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(runs)} planned runs to {args.manifest}")
    if not args.run:
        print("Plan only: no training processes started.")
        return
    for run in runs:
        output = ROOT / run["output_dir"]
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to mix experiments in non-empty directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        print(f"Starting {run['name']}: {run['steps']} steps, {run['tokens']:,} tokens", flush=True)
        with (output / "console.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(run["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        run["exit_code"] = completed.returncode
        run["status"] = "completed" if completed.returncode == 0 else "failed"
        metrics = output / "metrics.jsonl"
        if metrics.exists():
            records = [json.loads(line) for line in metrics.read_text().splitlines()]
            validation = [record for record in records if "valid_loss" in record]
            if validation:
                run["last_validation"] = validation[-1]
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if completed.returncode:
            console = (output / "console.log").read_text(encoding="utf-8")
            if "OutOfMemory" in console or "out of memory" in console:
                print(f"Stopped at memory limit. See {output / 'console.log'}", flush=True)
                raise SystemExit(1)
            if "Non-finite" not in console:
                print(f"Stopped at unexpected failure. See {output / 'console.log'}", flush=True)
                raise SystemExit(1)


if __name__ == "__main__":
    main()
