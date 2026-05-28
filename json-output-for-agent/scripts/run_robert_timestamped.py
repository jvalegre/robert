#!/usr/bin/env python3
"""Timestamped ROBERT run wrapper that does not change ROBERT output behavior.

This wrapper runs ROBERT exactly as usual from the repository root so standard
outputs are still produced in their normal locations (CURATE/, GENERATE/, etc.).
After ROBERT finishes, it copies those generated outputs into a timestamped run
archive under json-output-for-agent/runs/ for project-side tracking.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


PATH_OPTIONS = {"csv_name", "csv_test", "varfile", "params_dir"}
COPY_DIRS = ["CURATE", "GENERATE", "VERIFY", "PREDICT", "REPORT", "AQME", "EVALUATE"]
COPY_FILES = ["ROBERT_report.pdf", "report.css", "report_debug.txt"]


def _sanitize_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "run"


def _extract_option_value(args: List[str], option: str) -> Optional[str]:
    long_opt = f"--{option}"
    prefix = f"{long_opt}="
    for i, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
        if arg == long_opt and i + 1 < len(args):
            return args[i + 1]
    return None


def _absolutize_path_options(args: List[str], base_dir: Path) -> List[str]:
    out = list(args)
    i = 0
    while i < len(out):
        arg = out[i]
        if not arg.startswith("--"):
            i += 1
            continue

        if "=" in arg:
            key, value = arg[2:].split("=", 1)
            if key in PATH_OPTIONS and value not in {"", "None"}:
                out[i] = f"--{key}={(base_dir / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()}"
            i += 1
            continue

        key = arg[2:]
        if key in PATH_OPTIONS and i + 1 < len(out):
            val = out[i + 1]
            if val not in {"", "None"} and not val.startswith("--"):
                p = Path(val)
                out[i + 1] = str((base_dir / p).resolve() if not p.is_absolute() else p.resolve())
        i += 1

    return out


def _make_run_dir(runs_root: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = runs_root / f"{stamp}_{label}"
    candidate = base
    n = 2
    while candidate.exists():
        candidate = runs_root / f"{stamp}_{label}_{n}"
        n += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _copy_outputs(repo_root: Path, run_dir: Path) -> None:
    for name in COPY_DIRS:
        src = repo_root / name
        if src.exists() and src.is_dir():
            shutil.copytree(src, run_dir / name, dirs_exist_ok=True)

    for name in COPY_FILES:
        src = repo_root / name
        if src.exists() and src.is_file():
            shutil.copy2(src, run_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run ROBERT from repository root, then copy standard outputs into a "
            "timestamped archive under json-output-for-agent/runs/."
        )
    )
    parser.add_argument(
        "--wrapper-runs-root",
        default="json-output-for-agent/runs",
        help="Archive root relative to repo root or absolute path.",
    )
    parser.add_argument(
        "--wrapper-python",
        default=sys.executable,
        help="Python executable used to run 'python -m robert'.",
    )
    parser.add_argument(
        "--wrapper-label",
        default=None,
        help="Optional run label override. Default comes from --csv_name stem.",
    )
    parser.add_argument(
        "--wrapper-dry-run",
        action="store_true",
        help="Print planned actions without running ROBERT.",
    )

    known, robert_args = parser.parse_known_args()
    if robert_args and robert_args[0] == "--":
        robert_args = robert_args[1:]

    if not robert_args:
        print("x No ROBERT arguments were provided. Pass standard ROBERT args after wrapper args.")
        return 2

    invocation_cwd = Path.cwd().resolve()
    repo_root = Path(__file__).resolve().parents[2]

    csv_name_value = _extract_option_value(robert_args, "csv_name")
    label = known.wrapper_label
    if label is None:
        if not csv_name_value:
            print("x Could not infer run label because --csv_name was not provided.")
            print("  Provide --csv_name or set --wrapper-label explicitly.")
            return 2
        label = _sanitize_name(Path(csv_name_value).stem)

    runs_root = Path(known.wrapper_runs_root)
    if not runs_root.is_absolute():
        runs_root = (repo_root / runs_root).resolve()

    normalized_robert_args = _absolutize_path_options(robert_args, invocation_cwd)
    cmd = [known.wrapper_python, "-m", "robert", *normalized_robert_args]

    run_dir = _make_run_dir(runs_root, _sanitize_name(label))
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "repo_root": str(repo_root),
        "invocation_cwd": str(invocation_cwd),
        "run_dir": str(run_dir),
        "command": cmd,
        "copy_mode": (
            "ROBERT normal root outputs are preserved; this wrapper only duplicates "
            "generated artifacts into run_dir."
        ),
    }

    if known.wrapper_dry_run:
        metadata["dry_run"] = True
        (run_dir / "wrapper_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"o Dry run created archive folder: {run_dir}")
        print(f"o Planned command: {' '.join(cmd)}")
        return 0

    print(f"o Running ROBERT from repo root: {repo_root}")
    print(f"o Archive folder: {run_dir}")
    completed = subprocess.run(cmd, cwd=repo_root)

    _copy_outputs(repo_root, run_dir)
    metadata["return_code"] = completed.returncode
    (run_dir / "wrapper_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"o Copied generated outputs to: {run_dir}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
