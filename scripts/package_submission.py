"""Package source and small test fixtures, excluding datasets and environments."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission/code.zip")
    parser.add_argument("--verify", action="store_true", help="Run tests from a temporary extracted copy")
    args = parser.parse_args()
    files = []
    for folder in ("cs336_basics", "ablation_experiments", "tests", "scripts"):
        for path in sorted((ROOT / folder).rglob("*")):
            if not path.is_file() or any(part.startswith(".") or part == "__pycache__" for part in path.parts):
                continue
            if path.name in {"build_assignment_report.py", "requirements-report.txt"} and path.parent == ROOT / "scripts":
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            if path.stat().st_size > 10 * 1024 * 1024:
                raise ValueError(f"Unexpected large source/fixture: {path}")
            files.append(path)
    for name in ("pyproject.toml", "uv.lock", "README.md", "make_submission.sh"):
        files.append(ROOT / name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if bad := archive.testzip():
            raise ValueError(f"Zip integrity check failed: {bad}")
        names = archive.namelist()
        assert any(name.startswith("tests/fixtures/") for name in names)
        assert "cs336_basics/train.py" in names
        assert not any(name.startswith(("data/", "checkpoints/", ".venv")) for name in names)
    temporary.replace(args.output)
    args.output.with_suffix(".manifest.json").write_text(json.dumps({
        "files": names, "compressed_bytes": args.output.stat().st_size,
        "status": "source archive",
    }, indent=2), encoding="utf-8")
    print(f"Packaged and verified {len(names)} files: {args.output} ({args.output.stat().st_size:,} bytes)")
    if args.verify:
        with tempfile.TemporaryDirectory(prefix="cs336_submission_") as directory:
            with zipfile.ZipFile(args.output) as archive:
                archive.extractall(directory)
            environment = dict(os.environ, PYTHONPATH=directory, OMP_NUM_THREADS="1")
            with (ROOT / "results/package_verification.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run([
                    sys.executable, "-m", "pytest", "tests", "-q", "--timeout", "10",
                    f"--junitxml={ROOT / 'results/archive_test_results.xml'}",
                ], cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise SystemExit("Extracted archive tests failed; see results/package_verification.log")
            print("Extracted archive tests passed; see results/archive_test_results.xml")


if __name__ == "__main__":
    main()
