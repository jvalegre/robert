#!/usr/bin/env python
"""
Full ROBERT workflow via the Python API.

Runs CURATE, GENERATE (Bayesian hyperparameter optimization), VERIFY,
PREDICT, and REPORT; prints sklearn and ROBERT scores and artifact paths.

Usage (from repository root)::

    python Examples/API_workflow/robert_api_full_workflow.py

Requires WeasyPrint system libraries for PDF generation (see ROBERT docs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from robert import RobertModel  # noqa: E402

_CSV = _REPO / "tests" / "Robert_example.csv"
_WORKDIR = Path(__file__).resolve().parent / "robert_api_run"


def main() -> None:
    df = pd.read_csv(_CSV, encoding="utf-8")
    X = df.drop(columns=["Target_values"])
    y = df["Target_values"]
    n_fit = 25

    model = RobertModel(
        problem_type="reg",
        filter_mode="no_pfi",
        workdir=_WORKDIR,
        names="Name",
        report=True,
        plot_verbosity=2,
        model=["RF", "GB"],
        n_iter=2,
        init_points=2,
        repeat_kfolds=2,
        kfold=3,
        seed=42,
    )

    print("Fitting (CURATE → GENERATE → VERIFY → PREDICT → REPORT)...")
    model.fit(X.iloc[:n_fit], y.iloc[:n_fit])

    info = model.best_model_info()
    print(f"Best model: {info['model']}  descriptors: {len(info['descriptors'])}")

    X_test = X.iloc[n_fit:].drop_duplicates(subset=["Name"], keep="first")
    y_test = y.loc[X_test.index]
    preds = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    print(f"Holdout R² (sklearn): {r2:.3f}")
    print(f"Holdout predictions shape: {preds.shape}")

    scores = model.robert_scores()
    print(f"\nROBERT score ({scores['suffix']}): {scores['robert_score']}")
    print("Sub-scores:")
    for name, value in scores["components"].items():
        print(f"  - {name}: {value}")

    workdir = Path(model.workdir_)
    print("\nKey outputs:")
    for pattern in (
        "CURATE/*.png",
        "GENERATE/Heatmap*.png",
        "VERIFY/VERIFY_tests_*.png",
        "PREDICT/*.png",
        "ROBERT_report.pdf",
    ):
        matches = sorted(workdir.glob(pattern))
        for path in matches[:3]:
            print(f"  {path.relative_to(workdir)}")
        if len(matches) > 3:
            print(f"  ... ({len(matches)} files matching {pattern})")

    if scores["pdf_path"]:
        print(f"\nReport PDF: {scores['pdf_path']}")
    else:
        print("\nx  ROBERT_report.pdf was not created (WeasyPrint may be missing).")


if __name__ == "__main__":
    main()
