"""
Sklearn-style facade over the ROBERT CLI pipeline.

``RobertModel`` runs CURATE through PREDICT (optional REPORT). Point
predictions, uncertainty columns, and matplotlib handling are documented in
``docs/API/robert.api.rst``.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score

from robert.argument_parser import options_add, var_dict

_NAME_COL = "__robert_name__"
_DEFAULT_Y = "__robert_y__"


def _filter_robert_kwargs(kwargs: dict) -> dict:
    """Keep only keys accepted by ``set_options`` / ``load_variables``."""
    return {k: v for k, v in kwargs.items() if k in var_dict}


@contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@contextmanager
def _noninteractive_mpl():
    """Use Agg during ROBERT steps; restore the previous backend afterward."""
    prev = matplotlib.get_backend()
    was_interactive = matplotlib.is_interactive()
    matplotlib.use("Agg", force=True)
    plt.ioff()
    try:
        yield
    finally:
        matplotlib.use(prev, force=True)
        if was_interactive:
            plt.ion()


class _ParamsAdapter:
    """Minimal ``self`` for :func:`robert.utils.load_params` / ``model_adjust_params``."""

    __slots__ = ("args",)

    def __init__(self, seed: int, type_: str):
        self.args = options_add()
        self.args.seed = seed
        self.args.type = type_


def _suffix_title(filter_mode: str) -> str:
    if filter_mode == "pfi":
        return "PFI"
    if filter_mode == "no_pfi":
        return "No_PFI"
    raise ValueError("filter_mode must be 'pfi' or 'no_pfi'")


def _find_params_csv(best_subdir: Path) -> Path:
    if not best_subdir.is_dir():
        raise FileNotFoundError(str(best_subdir))
    csvs = sorted(
        p for p in best_subdir.glob("*.csv") if not p.name.endswith("_db.csv")
    )
    if len(csvs) != 1:
        raise RuntimeError(
            "Expected exactly one parameter CSV in "
            f"{best_subdir}, found: {[p.name for p in csvs]}"
        )
    return csvs[0]


def _resolve_prediction_id_column(
    result_df: pd.DataFrame, names_key: str, model_names: str
) -> str:
    """
    Column in the PREDICT external CSV used to align rows to the API input order.

    Prefer the API names column, then the name stored in GENERATE params, then a
    single case-insensitive match. Avoid assuming ``result_df.columns[0]`` is the id
    column (column order may change).
    """
    cols = list(result_df.columns)
    if names_key in result_df.columns:
        return names_key
    if model_names and model_names in result_df.columns:
        return model_names
    candidates = [c for c in (names_key, model_names) if c]
    for cand in candidates:
        cf = cand.casefold()
        matches = [c for c in cols if str(c).casefold() == cf]
        if len(matches) == 1:
            return str(matches[0])
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous id column for predictions: {matches!r} all match {cand!r} "
                f"(case-insensitive). Columns in CSV: {cols!r}"
            )
    raise RuntimeError(
        f"Could not find row id column for predictions (tried {names_key!r} and "
        f"{model_names!r}). Columns in CSV: {cols!r}"
    )


def _resolve_predict_csv(
    workdir: Path, pred_stem: str, model_code: str, suffix: str
) -> str:
    """Path to PREDICT output CSV for the external set (exact file, else sorted glob)."""
    csv_dir = workdir / "PREDICT" / "csv_test"
    exact = csv_dir / f"{pred_stem}_{model_code}_{suffix}.csv"
    if exact.is_file():
        return str(exact)
    pattern = str(csv_dir / f"{pred_stem}_{model_code}_{suffix}.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        alt_pat = str(csv_dir / f"*{pred_stem}*{suffix}.csv")
        matches = sorted(glob.glob(alt_pat))
    if not matches:
        raise RuntimeError(f"No prediction CSV found matching {pattern!r}")
    if len(matches) > 1:
        warnings.warn(
            "Multiple prediction CSVs matched; using the first after lexicographic sort: "
            f"{matches[0]!r}",
            UserWarning,
            stacklevel=3,
        )
    return matches[0]


class RobertModel(BaseEstimator):
    """
    High-level ROBERT workflow with ``fit``, ``predict``, and ``score``.

    Subclasses :class:`~sklearn.base.BaseEstimator` for ``get_params`` and
    ``set_params``. Preprocessing (CURATE encoding, ROBERT-internal
    ``StandardScaler`` on the design matrix) runs inside ROBERT; do not stack a
    separate ``StandardScaler`` on the same raw descriptors. Refitting with the same
    ``workdir`` leaves prior outputs on disk. ``fit`` and ``predict`` change the
    process working directory and are not safe to run concurrently on multiple
    instances in one process.

    :param problem_type: ``"reg"`` or ``"clas"``.
    :param filter_mode: ``"pfi"`` or ``"no_pfi"``; which ``GENERATE/Best_model``
        variant ``predict`` loads.
    :param workdir: Directory for ROBERT outputs, or ``None`` for a managed temp dir.
    :param names: Column in ``X`` used as row identifiers (CURATE ``names``), or ``None``.
    :param report: If ``True``, run REPORT after PREDICT during ``fit``.
    :param seed: Random seed forwarded to ROBERT, or ``None`` for package default.
    :param y_column: If ``fit(X)`` is called with ``y is None``, name of the target column
        in ``X`` (DataFrame only).
    :param kwargs: Additional ROBERT options (keys in ``robert.argument_parser.var_dict``),
        e.g. ``model``, ``n_iter``, ``plot_verbosity`` (``0``–``2``): ``0`` skips
        matplotlib artifacts; ``1`` keeps CURATE/GENERATE/VERIFY summary plots and main
        PREDICT result plots when ``predict_diagnostics`` is True; ``2`` additionally
        enables SHAP/PFI/Pearson/outlier/distribution diagnostics when
        ``predict_diagnostics`` is True. Regression uncertainty tuning includes
        ``conformal_enable``, ``conformal_calib_frac``, and ``conformal_coverage``.
        Top-k meta-model uncertainty (opt-in) uses ``uq_enable_meta``,
        ``uq_top_k_models``, and ``uq_model_weighting`` (``"score_weighted"`` or
        ``"uniform"``). Auto uncertainty (regression, opt-in) uses ``uq_auto_enable``,
        ``uq_auto_candidates``, ``uq_auto_scaler``, ``uq_auto_metric_weights``,
        ``uq_auto_min_samples``, ``uq_auto_random_state``, and ``uq_auto_clas_mode``
        (``"error"`` by default); ``return_uncertainty`` ``"auto"`` or
        ``"auto_decomposed"`` enables auto mode for that predict call.
    """

    def __init__(
        self,
        problem_type: Literal["reg", "clas"] = "reg",
        filter_mode: Literal["pfi", "no_pfi"] = "pfi",
        workdir: Optional[Union[str, Path]] = None,
        names: Optional[str] = None,
        report: bool = False,
        seed: Optional[int] = None,
        y_column: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        kwargs = dict(kwargs)
        if "type" in kwargs:
            warnings.warn(
                "Passing 'type' to RobertModel is deprecated; use 'problem_type' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if problem_type != "reg":
                raise ValueError(
                    "Pass only one of 'problem_type' or deprecated 'type'."
                )
            problem_type = kwargs.pop("type")  # type: ignore[assignment]
        if "filter" in kwargs:
            warnings.warn(
                "Passing 'filter' to RobertModel is deprecated; use 'filter_mode' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if filter_mode != "pfi":
                raise ValueError(
                    "Pass only one of 'filter_mode' or deprecated 'filter'."
                )
            filter_mode = kwargs.pop("filter")  # type: ignore[assignment]

        self.problem_type = problem_type
        self.filter_mode = filter_mode
        self._user_names_col = names
        self.run_report = report
        self.y_column_init = y_column
        self._workdir_raw: Optional[Union[str, Path]] = workdir
        self._seed_explicit = seed

        raw_kw = _filter_robert_kwargs(kwargs)
        if seed is not None:
            raw_kw["seed"] = seed
        raw_kw["type"] = problem_type
        self._rob_kwargs = raw_kw

        self._managed_temp = workdir is None
        self.workdir_: Optional[Path] = (
            Path(workdir).resolve() if workdir is not None else None
        )
        self._tempdir_obj: Optional[tempfile.TemporaryDirectory[str]] = None

        self.is_fitted_ = False
        self.y_col_: str = ""
        self.X_columns_: list[str] = []
        self.model_data_: dict[str, Any] = {}
        self.names_col_: str = ""
        self.n_features_in_: int = 0
        self.feature_names_in_: Optional[np.ndarray] = None
        self._fit_input_was_dataframe = False

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Estimator and ROBERT option names from ``__init__`` (no nested estimators)."""
        _ = deep
        out: dict[str, Any] = {
            "problem_type": self.problem_type,
            "filter_mode": self.filter_mode,
            "workdir": self._workdir_raw,
            "names": self._user_names_col,
            "report": self.run_report,
            "seed": self._seed_explicit,
            "y_column": self.y_column_init,
        }
        for k, v in self._rob_kwargs.items():
            if k == "type":
                continue
            out[k] = v
        return out

    def set_params(self, **params: Any) -> "RobertModel":
        """Update ``RobertModel`` fields and ROBERT keys accepted by ``var_dict``."""
        if not params:
            return self
        special = {
            "problem_type",
            "filter_mode",
            "workdir",
            "names",
            "report",
            "seed",
            "y_column",
        }
        for key in list(params):
            if key not in special and key not in var_dict:
                raise ValueError(
                    f"RobertModel got invalid parameter {key!r}. "
                    "Use problem_type, filter_mode, workdir, names, report, seed, y_column, "
                    "or a key from robert.argument_parser.var_dict."
                )
        for key in list(params):
            if key not in special:
                continue
            val = params.pop(key)
            if key == "problem_type":
                self.problem_type = val
                self._rob_kwargs["type"] = val
            elif key == "filter_mode":
                self.filter_mode = val
            elif key == "workdir":
                self._workdir_raw = val
                self.workdir_ = Path(val).resolve() if val is not None else None
                self._managed_temp = val is None
                self._tempdir_obj = None
            elif key == "names":
                self._user_names_col = val
            elif key == "report":
                self.run_report = val
            elif key == "seed":
                self._seed_explicit = val
                if val is not None:
                    self._rob_kwargs["seed"] = val
                else:
                    self._rob_kwargs.pop("seed", None)
            elif key == "y_column":
                self.y_column_init = val
        for k, v in params.items():
            self._rob_kwargs[k] = v
            if k == "type":
                self.problem_type = v  # noqa: PLW2901
        return self

    def __enter__(self) -> "RobertModel":
        self._ensure_workdir()
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        self.cleanup()
        return False

    def cleanup(self) -> None:
        """Remove managed temporary workdir, if any. Safe to call multiple times."""
        if self._tempdir_obj is not None:
            self._tempdir_obj.cleanup()
            self._tempdir_obj = None
            self.workdir_ = None
        self._managed_temp = self._workdir_raw is None

    def _ensure_workdir(self) -> Path:
        if self.workdir_ is not None:
            self.workdir_.mkdir(parents=True, exist_ok=True)
            return self.workdir_
        if self._tempdir_obj is None:
            self._tempdir_obj = tempfile.TemporaryDirectory(prefix="robert_api_")
            self.workdir_ = Path(self._tempdir_obj.name).resolve()
            self._managed_temp = True
        assert self.workdir_ is not None
        self.workdir_.mkdir(parents=True, exist_ok=True)
        return self.workdir_

    def _coerce_xy(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray, list]],
        names: Optional[str],
    ) -> Tuple[pd.DataFrame, pd.Series, str]:
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError("X must be 2-D")
            X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            raise TypeError("X must be a pandas DataFrame or numpy.ndarray")

        names_col = names or self._user_names_col
        if names_col is not None and names_col not in X_df.columns:
            raise ValueError(f"names column {names_col!r} not found in X")

        if y is None:
            ykey = self.y_column_init
            if ykey is None or ykey not in X_df.columns:
                raise ValueError(
                    "y is None but no valid y_column in X; pass y or set y_column="
                )
            y_series = X_df[ykey].copy()
            X_df = X_df.drop(columns=[ykey])
        else:
            y_series = pd.Series(np.asarray(y).ravel(), index=X_df.index)
            if isinstance(y, pd.Series) and y.name is not None and str(y.name).strip():
                y_col_guess = str(y.name)
            else:
                y_col_guess = self.y_column_init or _DEFAULT_Y
            y_series.name = y_col_guess

        if names_col is None:
            X_df = X_df.copy()
            X_df.insert(0, _NAME_COL, X_df.index.astype(str))
            names_col = _NAME_COL

        y_col = str(y_series.name) if y_series.name is not None else _DEFAULT_Y
        y_series.name = y_col
        return X_df, y_series, names_col

    def _build_train_frame(
        self, X_df: pd.DataFrame, y_series: pd.Series
    ) -> pd.DataFrame:
        out = X_df.copy()
        out[y_series.name] = y_series.values
        return out

    def _read_model_snapshot(self, workdir: Path) -> dict[str, Any]:
        sub = "PFI" if self.filter_mode == "pfi" else "No_PFI"
        from robert.utils import load_params, path_generate_best_model

        folder = path_generate_best_model(workdir, sub)
        params_path = _find_params_csv(folder)
        seed = int(self._rob_kwargs.get("seed", var_dict["seed"]))
        adapter = _ParamsAdapter(seed, self.problem_type)
        return load_params(adapter, str(params_path))

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray, list]] = None,
        names: Optional[str] = None,
        **fit_params: Any,
    ) -> "RobertModel":
        _ = fit_params
        self._fit_input_was_dataframe = isinstance(X, pd.DataFrame)
        workdir = self._ensure_workdir()
        X_df, y_series, names_col = self._coerce_xy(X, y, names)
        train_csv = workdir / "_robert_train.csv"
        train_rel = train_csv.name
        y_col = str(y_series.name)

        train_frame = self._build_train_frame(X_df, y_series)
        train_frame.to_csv(train_csv, index=False)

        base = dict(self._rob_kwargs)
        base["command_line"] = False
        base["csv_test"] = ""

        from robert.curate import curate
        from robert.generate import generate
        from robert.predict import predict as predict_module
        from robert.report import report as report_module
        from robert.verify import verify
        from robert.utils import path_generate_best_model

        with _noninteractive_mpl(), _chdir(workdir):
            curate(
                csv_name=train_rel,
                y=y_col,
                names=names_col,
                **base,
            )
            generate(
                csv_name="",
                y="",
                discard=[],
                **base,
            )
            verify(**base)
            predict_module(**base)
            if self.run_report:
                report_module(**base)

        best_sub = path_generate_best_model(
            workdir, "PFI" if self.filter_mode == "pfi" else "No_PFI"
        )
        if self.filter_mode == "pfi" and not best_sub.is_dir():
            raise RuntimeError(
                "PFI model folder missing after fit. "
                "Try filter_mode='no_pfi' or enable pfi_filter in ROBERT options."
            )
        if self.filter_mode == "no_pfi" and not best_sub.is_dir():
            raise RuntimeError("No_PFI model folder missing after fit.")

        model_data = self._read_model_snapshot(workdir)
        self.model_data_ = model_data
        self.y_col_ = str(model_data["y"])
        descs = list(model_data["X_descriptors"])
        self.X_columns_ = descs
        self.n_features_in_ = len(descs)
        if self._fit_input_was_dataframe:
            self.feature_names_in_ = np.asarray(descs, dtype=object)
        else:
            self.feature_names_in_ = None
        model_names = str(model_data.get("names") or "")
        if model_names != names_col:
            if (
                model_names
                and names_col
                and model_names.casefold() == names_col.casefold()
            ):
                warnings.warn(
                    f"Names column casing normalized to saved model column {model_names!r} "
                    f"(was {names_col!r}).",
                    UserWarning,
                    stacklevel=2,
                )
                names_col = model_names
            else:
                raise RuntimeError(
                    "Names column mismatch between training data "
                    f"({names_col!r}) and GENERATE params ({model_names!r}). "
                    "Refit with a clean workdir or align the names column with the model."
                )
        self.names_col_ = names_col
        self.is_fitted_ = True
        return self

    def best_model_info(self) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before best_model_info.")
        md = self.model_data_
        return {
            "model": md["model"],
            "params": md["params"],
            "descriptors": md["X_descriptors"],
            "filter_mode": self.filter_mode,
            "y": md["y"],
        }

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_std: bool = False,
        return_uncertainty: Literal[
            False,
            "cv_sd",
            "conformal",
            "both",
            "meta",
            "total",
            "decomposed",
            "auto",
            "auto_decomposed",
        ] = False,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, dict],
    ]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before predict.")
        workdir = self.workdir_
        assert workdir is not None

        if return_uncertainty is not False:
            umode: Literal[
                False,
                "cv_sd",
                "conformal",
                "both",
                "meta",
                "total",
                "decomposed",
                "auto",
                "auto_decomposed",
            ] = return_uncertainty
            if return_std:
                warnings.warn(
                    "return_uncertainty is set; return_std is ignored.",
                    UserWarning,
                    stacklevel=2,
                )
        elif return_std:
            umode = "cv_sd"
        else:
            umode = False

        descriptors = list(self.model_data_["X_descriptors"])
        y_target = str(self.model_data_["y"])
        names_key = self.names_col_

        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError("X must be 2-D")
            if X.shape[1] != len(descriptors):
                raise ValueError(
                    f"X has {X.shape[1]} columns but model expects "
                    f"{len(descriptors)} descriptors"
                )
            X_df = pd.DataFrame(X, columns=descriptors, copy=False)
        else:
            X_df = X.copy()
            missing = [c for c in descriptors if c not in X_df.columns]
            if missing:
                tail = "..." if len(missing) > 10 else ""
                raise ValueError(f"Missing descriptor columns: {missing[:10]!r}{tail}")

        pred_id = uuid.uuid4().hex[:12]
        pred_name = f"_robert_predict_{pred_id}.csv"
        pred_path = workdir / pred_name

        if names_key and names_key in X_df.columns:
            X_out = X_df[[names_key] + descriptors].copy()
        else:
            X_out = X_df[descriptors].copy()
            X_out.insert(0, names_key or _NAME_COL, X_out.index.astype(str))

        order_keys = X_out.iloc[:, 0].astype(str)
        if order_keys.duplicated().any():
            raise ValueError(
                "Duplicate values in the names column; cannot align predictions to rows."
            )
        X_out.to_csv(pred_path, index=False)

        base = dict(self._rob_kwargs)
        base["command_line"] = False
        base["csv_test"] = pred_name
        base["params_dir"] = "GENERATE/Best_model"
        base["names"] = self.names_col_
        base["predict_diagnostics"] = False
        if umode in ("auto", "auto_decomposed"):
            if self.problem_type != "reg":
                raise ValueError(
                    "return_uncertainty='auto' or 'auto_decomposed' is only "
                    "supported for problem_type='reg'."
                )
            base["uq_auto_enable"] = True

        from robert.predict import predict as predict_module

        with _noninteractive_mpl(), _chdir(workdir):
            predict_module(**base)

        suffix = _suffix_title(self.filter_mode)
        model_code = str(self.model_data_["model"])
        pred_stem = Path(pred_name).stem
        csv_path = _resolve_predict_csv(workdir, pred_stem, model_code, suffix)

        result_df = pd.read_csv(csv_path, encoding="utf-8")
        pred_col = f"{y_target}_pred"
        sd_col = f"{y_target}_pred_sd"
        hw_col = f"{y_target}_pred_conformal_hw"
        uq_model_col = f"{y_target}_pred_uq_model"
        uq_meta_col = f"{y_target}_pred_uq_meta"
        uq_total_col = f"{y_target}_pred_uq_total"
        uq_auto_col = f"{y_target}_pred_uq_auto"
        uq_auto_src_col = f"{y_target}_pred_uq_auto_source"
        if pred_col not in result_df.columns:
            raise RuntimeError(f"Column {pred_col!r} missing in {csv_path}")

        model_names = str(self.model_data_.get("names") or "")
        name_col_result = _resolve_prediction_id_column(
            result_df, names_key, model_names
        )
        aligned = result_df.set_index(
            result_df[name_col_result].astype(str), drop=False
        )
        try:
            ordered = aligned.reindex(order_keys.values)
        except ValueError as err:
            raise RuntimeError(
                "Could not align predictions to input row names; duplicate names?"
            ) from err
        if ordered[pred_col].isna().any():
            missing_n = int(ordered[pred_col].isna().sum())
            raise RuntimeError(
                f"Prediction CSV row count or names mismatch ({missing_n} missing rows)."
            )

        y_pred = ordered[pred_col].to_numpy()
        try:
            os.remove(pred_path)
        except OSError:
            pass

        if umode is False:
            return y_pred

        if umode in ("cv_sd", "both"):
            if sd_col not in result_df.columns:
                raise RuntimeError(f"Column {sd_col!r} missing in {csv_path}")
            y_sd = ordered[sd_col].to_numpy(dtype=float)
        if umode in ("conformal", "both"):
            if self.problem_type != "reg":
                raise ValueError(
                    "return_uncertainty='conformal' or 'both' is only supported "
                    "for problem_type='reg'."
                )
            if hw_col not in result_df.columns:
                raise RuntimeError(f"Column {hw_col!r} missing in {csv_path}")
            y_hw = ordered[hw_col].to_numpy(dtype=float)
            if np.all(np.isnan(y_hw)):
                raise RuntimeError(
                    f"Column {hw_col!r} has no finite values; disable conformal "
                    "or use a larger training set."
                )
        if umode in ("auto", "auto_decomposed"):
            if uq_auto_col not in result_df.columns:
                raise RuntimeError(
                    f"Column {uq_auto_col!r} missing in {csv_path}; ensure "
                    "uq_auto_enable=True or use return_uncertainty='auto'."
                )
            y_uq_auto = ordered[uq_auto_col].to_numpy(dtype=float)
            if not (np.isfinite(y_uq_auto).all() and (y_uq_auto >= 0).all()):
                raise RuntimeError(
                    f"Column {uq_auto_col!r} has invalid uncertainty values."
                )
            auto_metadata: dict[str, Any] = {}
            meta_path = workdir / "PREDICT" / "uq_auto_metadata.json"
            if meta_path.is_file():
                with meta_path.open(encoding="utf-8") as fh:
                    auto_metadata = json.load(fh)
            elif uq_auto_src_col in result_df.columns:
                src_vals = ordered[uq_auto_src_col].dropna().unique()
                if len(src_vals):
                    auto_metadata["selected"] = str(src_vals[0])

        if umode in ("meta", "total", "decomposed"):
            if not bool(self._rob_kwargs.get("uq_enable_meta", False)):
                raise ValueError(
                    "return_uncertainty='meta', 'total', or 'decomposed' requires "
                    "uq_enable_meta=True when constructing RobertModel."
                )
            for col in (uq_model_col, uq_meta_col, uq_total_col):
                if col not in result_df.columns:
                    raise RuntimeError(
                        f"Column {col!r} missing in {csv_path}; refit with "
                        "uq_enable_meta=True or run predict after enabling meta UQ."
                    )
            y_uq_model = ordered[uq_model_col].to_numpy(dtype=float)
            y_uq_meta = ordered[uq_meta_col].to_numpy(dtype=float)
            y_uq_total = ordered[uq_total_col].to_numpy(dtype=float)
            if not (np.isfinite(y_uq_total).all() and (y_uq_total >= 0).all()):
                raise RuntimeError(
                    f"Column {uq_total_col!r} has invalid uncertainty values."
                )

        if umode == "cv_sd":
            return y_pred, y_sd
        if umode == "conformal":
            return y_pred, y_hw
        if umode == "both":
            return y_pred, y_sd, y_hw
        if umode == "auto":
            return y_pred, y_uq_auto
        if umode == "auto_decomposed":
            return y_pred, y_uq_auto, auto_metadata
        if umode == "meta":
            return y_pred, y_uq_meta
        if umode == "total":
            return y_pred, y_uq_total
        return y_pred, y_uq_model, y_uq_meta, y_uq_total

    def robert_scores(
        self,
        suffix: Optional[Literal["No PFI", "PFI"]] = None,
    ) -> dict[str, Any]:
        """
        Return the ROBERT report score and sub-scores from VERIFY/PREDICT outputs.

        Requires a prior :meth:`fit` that ran VERIFY and PREDICT (and REPORT if a
        PDF is expected). Reads ``*_data.dat`` files in :attr:`workdir_`.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit before robert_scores.")
        workdir = self.workdir_
        if workdir is None:
            raise RuntimeError("workdir is not set.")

        if suffix is None:
            suffix = "PFI" if self.filter_mode == "pfi" else "No PFI"

        from robert.report_utils import calc_score, repro_info

        modules = ["CURATE", "GENERATE", "VERIFY", "PREDICT"]
        with _chdir(workdir):
            _, _, _, _, _, dat_files = repro_info(modules)
            if "PREDICT" not in dat_files or "VERIFY" not in dat_files:
                raise RuntimeError(
                    "PREDICT/VERIFY outputs missing in workdir; "
                    "run fit() with the full pipeline first."
                )
            data_score: dict[str, Any] = {}
            data_score = calc_score(dat_files, suffix, self.problem_type, data_score)

        score_key = f"robert_score_{suffix}"
        if self.problem_type == "reg":
            component_keys = [
                "cv_score_combined",
                "test_score_combined",
                "cv_sd_score",
                "diff_scaled_rmse_score",
                "flawed_mod_score",
                "sorted_cv_score",
            ]
        else:
            component_keys = [
                "cv_score_combined",
                "test_score_combined",
                "flawed_mod_score",
                "sorted_cv_score",
                "diff_mcc_score",
                "descp_score",
            ]
        components = {
            key: data_score.get(f"{key}_{suffix}", 0) for key in component_keys
        }
        pdf_path = workdir / "ROBERT_report.pdf"
        return {
            "suffix": suffix,
            "robert_score": int(data_score.get(score_key, 0)),
            "components": components,
            "pdf_path": str(pdf_path) if pdf_path.is_file() else None,
        }

    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, list],
    ) -> float:
        y_true = np.asarray(y).ravel()
        y_hat = np.asarray(self.predict(X, return_std=False)).ravel()
        if self.problem_type == "reg":
            return float(r2_score(y_true, y_hat))
        return float(
            accuracy_score(
                np.round(y_true).astype(int),
                np.round(y_hat).astype(int),
            )
        )
