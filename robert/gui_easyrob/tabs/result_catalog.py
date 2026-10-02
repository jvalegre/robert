"""Discover model-specific ROBERT result files for the GUI."""

from dataclasses import dataclass
from pathlib import Path
import re


VARIANTS = ("No_PFI", "PFI")


def _model_and_variant(stem):
    for variant in VARIANTS:
        suffix = f"_{variant}"
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[:-len(suffix)], variant
    return None


@dataclass(frozen=True)
class ResultCatalog:
    """Describe the result files for a ROBERT run."""

    root: Path
    models: tuple[str, ...]
    best_by_variant: dict[str, str]
    is_all_models: bool

    @classmethod
    def discover(cls, run_dir):
        """Find available model variants and ROBERT's selected best models."""
        root = Path(run_dir)
        model_variants = {}
        for path in (root / "PREDICT").glob("*.csv"):
            if path.name.startswith("Results_boundary_williams_"):
                continue
            parsed = _model_and_variant(path.stem)
            if parsed:
                model, variant = parsed
                model_variants.setdefault(model, set()).add(variant)
        for path in (root / "REPORT_models").glob("ROBERT_report_*.pdf"):
            parsed = _model_and_variant(path.stem.removeprefix("ROBERT_report_"))
            if parsed:
                model, variant = parsed
                model_variants.setdefault(model, set()).add(variant)

        all_models = (
            any((root / "REPORT_models").glob("ROBERT_report_*.pdf"))
            or (root / "GENERATE" / "All_models").is_dir()
        )

        best = {}
        if all_models:
            # --all_models: report.py's organize_all_models_pdfs() already picked a single
            # overall-best (model, variant) combining Interpolation and Boundary robustness
            # together, and copied only that one PDF back into the working directory - read
            # that directly instead of re-deriving a pick here. GENERATE/Best_model/{variant}
            # is a DIFFERENT, older selection (RMSE-only, resolved independently per variant),
            # which can disagree with the actual kept PDF and even differ between variants -
            # showing both as "the best" is exactly the confusing, inconsistent result this
            # replaces
            for path in sorted(root.glob("ROBERT_report_*.pdf")):
                parsed = _model_and_variant(path.stem.removeprefix("ROBERT_report_"))
                if parsed:
                    model, variant = parsed
                    best[variant] = model
        else:
            # single-model run (no --all_models): both No_PFI and PFI are the same model,
            # read straight from GENERATE/Best_model/{variant} as before
            best_root = root / "GENERATE" / "Best_model"
            for variant in VARIANTS:
                candidates = sorted(
                    path for path in (best_root / variant).glob("*.csv")
                    if not path.stem.endswith("_db")
                )
                if candidates:
                    name = candidates[0].stem
                    best[variant] = name.removesuffix("_PFI") if variant == "PFI" else name
                else:
                    for path in sorted(root.glob("ROBERT_report_*.pdf")):
                        parsed = _model_and_variant(path.stem.removeprefix("ROBERT_report_"))
                        if parsed and parsed[1] == variant:
                            best[variant] = parsed[0]
                            break
                if variant not in best:
                    best[variant] = next(
                        (model for model in sorted(model_variants)
                         if variant in model_variants[model]),
                        "",
                    )
        best = {variant: model for variant, model in best.items() if model}
        return cls(root, tuple(sorted(model_variants)), best, all_models)

    def selected_variants(self, model=None):
        """Map each available PFI variant to its selected model."""
        if model is None:
            return self.best_by_variant
        return {variant: model for variant in VARIANTS if (
            self.root / "PREDICT" / f"{model}_{variant}.csv"
        ).is_file() or (
            self.root / "REPORT_models" / f"ROBERT_report_{model}_{variant}.pdf"
        ).is_file()}

    def report_paths(self, model=None):
        """Return the default or selected model reports."""
        if not self.is_all_models:
            return sorted(self.root.glob("ROBERT_report*.pdf"))
        selected = self.selected_variants(model)
        return [
            path for variant in VARIANTS
            if (name := selected.get(variant))
            if (path := self.root / "REPORT_models" / f"ROBERT_report_{name}_{variant}.pdf").is_file()
        ]

    def internal_predictions(self, model=None):
        """Return prediction CSVs for the selected model variants."""
        selected = self.selected_variants(model)
        return {
            variant: path for variant in VARIANTS
            if (name := selected.get(variant))
            if (path := self.root / "PREDICT" / f"{name}_{variant}.csv").is_file()
        }

    def external_predictions(self, model=None):
        """Return optional external test CSVs for the selected variants."""
        selected = self.selected_variants(model)
        paths = sorted((self.root / "PREDICT" / "csv_test").glob("*.csv"))
        results = {}
        for variant, name in selected.items():
            match = next(
                (path for path in paths if path.stem.endswith(f"_{name}_{variant}")),
                None,
            )
            if match is not None:
                results[variant] = match
        return results

    def plot_paths(self, model=None):
        """List internal and optional external CSVs in display order."""
        internal = self.internal_predictions(model)
        external = self.external_predictions(model)
        return [*internal.values(), *external.values()]

    def include_image(self, path, model=None):
        """Keep shared figures and figures matching the selected model variants."""
        if not self.is_all_models:
            return True
        selected = self.selected_variants(model)
        matched = [
            (name, variant)
            for name in self.models
            for variant in VARIANTS
            if re.search(rf"(?:^|_){re.escape(name)}_{variant}(?:_|$)", path.stem)
        ]
        return not matched or any(selected.get(variant) == name for name, variant in matched)
