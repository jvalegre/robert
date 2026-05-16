"""
Automatic uncertainty selection and calibration for regression.

Evaluates multiple uncertainty candidates on out-of-fold training residuals,
fits optional global multiplicative scalers, and selects the best-calibrated
candidate for deployment on all prediction splits. User-facing API details are
in ``docs/API/robert.api.rst``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split

# Candidate identifiers (stable strings stored in CSV metadata).
CANDIDATE_CV_SD = "cv_sd"
CANDIDATE_CONFORMAL = "conformal"
CANDIDATE_META_TOTAL = "meta_total"

DEFAULT_CANDIDATES = (CANDIDATE_CV_SD, CANDIDATE_CONFORMAL, CANDIDATE_META_TOTAL)
DEFAULT_METRIC_WEIGHTS = {"coverage": 1.0, "sharpness": 0.25, "nll": 0.5}
TIE_BREAK_ORDER = (CANDIDATE_CV_SD, CANDIDATE_CONFORMAL, CANDIDATE_META_TOTAL)

VALID_SCALERS = ("none", "global_multiplicative", "isotonic")


def _as_float_array(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def _normalize_metric_weights(weights: Optional[Mapping[str, float]]) -> Dict[str, float]:
    base = dict(DEFAULT_METRIC_WEIGHTS)
    if weights is not None:
        for key in base:
            if key in weights:
                base[key] = float(weights[key])
    total = sum(base.values())
    if total <= 0:
        return dict(DEFAULT_METRIC_WEIGHTS)
    return {k: v / total for k, v in base.items()}


def fit_uncertainty_scaler(
    method: str,
    u_raw: np.ndarray,
    abs_residuals: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit a post-hoc scaler mapping raw uncertainty to calibrated sigma-like values.

    Parameters
    ----------
    method
        ``none``, ``global_multiplicative``, or ``isotonic``.
    u_raw, abs_residuals
        Aligned nonnegative uncertainty and absolute residual arrays.
    """
    method = str(method).lower()
    if method not in VALID_SCALERS:
        raise ValueError(
            f"Unknown uq_auto_scaler {method!r}; expected one of {VALID_SCALERS}."
        )

    u = np.maximum(_as_float_array(u_raw), 0.0)
    r = np.maximum(_as_float_array(abs_residuals), 0.0)
    mask = np.isfinite(u) & np.isfinite(r)
    u, r = u[mask], r[mask]

    if method == "none":
        return {"method": "none", "alpha": 1.0}

    if u.size < 2:
        return {"method": method, "alpha": 1.0}

    if method == "global_multiplicative":
        denom = float(np.dot(u, u))
        if denom <= 1e-12:
            alpha = 1.0
        else:
            alpha = float(np.dot(u, r) / denom)
        alpha = max(alpha, 1e-6)
        return {"method": "global_multiplicative", "alpha": alpha}

    # isotonic: piecewise-linear map on sorted unique u values
    order = np.argsort(u)
    u_sorted = u[order]
    r_sorted = r[order]
    # PAV-style simple isotonic: cumulative max of running medians in bins
    uniq_u, inv = np.unique(u_sorted, return_inverse=True)
    r_mean = np.zeros_like(uniq_u)
    counts = np.zeros_like(uniq_u)
    for idx, val in enumerate(r_sorted):
        b = inv[idx]
        r_mean[b] += val
        counts[b] += 1
    r_mean = r_mean / np.maximum(counts, 1)
    # enforce monotonicity in u
    for i in range(1, len(r_mean)):
        r_mean[i] = max(r_mean[i], r_mean[i - 1])
    return {
        "method": "isotonic",
        "knots_u": uniq_u.tolist(),
        "knots_r": r_mean.tolist(),
    }


def apply_uncertainty_scaler(
    method: str,
    u_raw: np.ndarray,
    params: Mapping[str, Any],
) -> np.ndarray:
    """Apply fitted scaler parameters to raw uncertainty."""
    u = np.maximum(_as_float_array(u_raw), 0.0)
    method = str(params.get("method", method)).lower()

    if method == "none":
        return u

    if method == "global_multiplicative":
        alpha = float(params.get("alpha", 1.0))
        return np.maximum(u * alpha, 0.0)

    if method == "isotonic":
        knots_u = np.asarray(params.get("knots_u", []), dtype=float)
        knots_r = np.asarray(params.get("knots_r", []), dtype=float)
        if knots_u.size == 0:
            return u
        return np.maximum(np.interp(u, knots_u, knots_r, left=knots_r[0], right=knots_r[-1]), 0.0)

    raise ValueError(f"Unknown scaler method in params: {method!r}")


def _coverage_error(abs_resid: np.ndarray, sigma: np.ndarray, coverage: float) -> float:
    sigma = np.maximum(sigma, 1e-12)
    # Gaussian half-width factor for symmetric interval
    alpha = 1.0 - coverage
    z = float(norm.ppf(1.0 - alpha / 2.0))
    inside = abs_resid <= z * sigma
    emp_cov = float(np.mean(inside)) if inside.size else 0.0
    return abs(emp_cov - coverage)


def _gaussian_nll(abs_resid: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(sigma, 1e-12)
    # NLL for Laplace-like on abs residual under Gaussian proxy
    var = sigma ** 2
    return float(np.mean(0.5 * np.log(2.0 * np.pi * var) + 0.5 * (abs_resid ** 2) / var))


def _sharpness(sigma: np.ndarray) -> float:
    return float(np.mean(np.maximum(sigma, 0.0)))


def score_uncertainty_candidate(
    u_scaled: np.ndarray,
    abs_resid: np.ndarray,
    coverage: float,
    metric_weights: Optional[Mapping[str, float]] = None,
) -> float:
    """
    Lower is better. Composite of coverage error, mean width, and Gaussian NLL.
    """
    w = _normalize_metric_weights(metric_weights)
    u_s = np.maximum(_as_float_array(u_scaled), 1e-12)
    r = _as_float_array(abs_resid)
    cov_err = _coverage_error(r, u_s, coverage)
    sharp = _sharpness(u_s)
    nll = _gaussian_nll(r, u_s)
    return w["coverage"] * cov_err + w["sharpness"] * sharp + w["nll"] * nll


def _oof_mean_train(Xy_data: Mapping[str, Any]) -> np.ndarray:
    preds_all = Xy_data.get("y_pred_train_all", [])
    return np.array([float(np.mean(p)) if len(p) else np.nan for p in preds_all], dtype=float)


def _train_abs_residuals(Xy_data: Mapping[str, Any]) -> np.ndarray:
    y_true = _as_float_array(Xy_data["y_train"])
    y_oof = _oof_mean_train(Xy_data)
    return np.abs(y_true - y_oof)


def _raw_candidate_train(
    candidate: str,
    Xy_data: Mapping[str, Any],
) -> Optional[np.ndarray]:
    n = len(Xy_data["y_train"])
    if candidate == CANDIDATE_CV_SD:
        key = "y_pred_train_uq_model"
        if key in Xy_data:
            return _as_float_array(Xy_data[key])
        if "y_pred_train_sd" in Xy_data:
            return _as_float_array(Xy_data["y_pred_train_sd"])
        return None
    if candidate == CANDIDATE_CONFORMAL:
        hw = float(Xy_data.get("conformal_half_width", float("nan")))
        if not np.isfinite(hw) or hw < 0:
            return None
        return np.full(n, hw, dtype=float)
    if candidate == CANDIDATE_META_TOTAL:
        key = "y_pred_train_uq_total"
        if key not in Xy_data:
            return None
        return _as_float_array(Xy_data[key])
    return None


def _raw_candidate_split(
    candidate: str,
    split: str,
    Xy_data: Mapping[str, Any],
) -> Optional[np.ndarray]:
    if split == "train":
        n = len(Xy_data["y_train"])
    elif split == "test":
        n = len(Xy_data["y_test"])
    elif split == "external":
        if "X_external" not in Xy_data:
            return None
        n = len(Xy_data["X_external"])
    else:
        return None

    if candidate == CANDIDATE_CV_SD:
        uq_key = f"y_pred_{split}_uq_model"
        sd_key = f"y_pred_{split}_sd"
        if uq_key in Xy_data:
            return _as_float_array(Xy_data[uq_key])
        if sd_key in Xy_data:
            return _as_float_array(Xy_data[sd_key])
        return None
    if candidate == CANDIDATE_CONFORMAL:
        hw = float(Xy_data.get("conformal_half_width", float("nan")))
        if not np.isfinite(hw) or hw < 0:
            return None
        return np.full(n, hw, dtype=float)
    if candidate == CANDIDATE_META_TOTAL:
        key = f"y_pred_{split}_uq_total"
        if key not in Xy_data:
            return None
        return _as_float_array(Xy_data[key])
    return None


def _available_candidates(
    candidates: Sequence[str],
    Xy_data: Mapping[str, Any],
    problem_type: str,
) -> List[str]:
    reg = problem_type.lower() == "reg"
    out: List[str] = []
    for cand in candidates:
        if cand == CANDIDATE_CONFORMAL and not reg:
            continue
        raw = _raw_candidate_train(cand, Xy_data)
        if raw is None or not np.isfinite(raw).any():
            continue
        out.append(cand)
    return out


def evaluate_uq_candidates(
    Xy_data: Mapping[str, Any],
    args: Any,
    problem_type: str,
) -> Dict[str, Any]:
    """
    Score candidates on training OOF residuals with an inner hold-out split.

    Returns dict with keys: selected, scaler_params, candidate_scores, coverage, n_eval.
    """
    candidates_cfg = getattr(args, "uq_auto_candidates", None) or list(DEFAULT_CANDIDATES)
    if isinstance(candidates_cfg, str):
        candidates_cfg = [c.strip() for c in candidates_cfg.split(",") if c.strip()]

    scaler_method = str(getattr(args, "uq_auto_scaler", "global_multiplicative"))
    coverage = float(getattr(args, "conformal_coverage", 0.9))
    min_samples = int(getattr(args, "uq_auto_min_samples", 12))
    seed = int(getattr(args, "uq_auto_random_state", getattr(args, "seed", 0)))
    metric_weights = getattr(args, "uq_auto_metric_weights", None)

    abs_resid = _train_abs_residuals(Xy_data)
    available = _available_candidates(candidates_cfg, Xy_data, problem_type)

    if not available:
        raise RuntimeError(
            "Auto uncertainty mode found no valid candidates on training data."
        )

    n = abs_resid.size
    if n < min_samples:
        warnings.warn(
            f"Training size {n} < uq_auto_min_samples={min_samples}; "
            "using unscaled best-effort selection.",
            UserWarning,
            stacklevel=2,
        )

    candidate_scores: Dict[str, float] = {}
    scaler_by_candidate: Dict[str, Dict[str, Any]] = {}

    for cand in available:
        u_raw = _raw_candidate_train(cand, Xy_data)
        if u_raw is None:
            continue
        if n >= max(min_samples, 8):
            idx = np.arange(n)
            fit_ix, eval_ix = train_test_split(
                idx,
                test_size=0.25,
                random_state=seed,
                shuffle=True,
            )
            params = fit_uncertainty_scaler(
                scaler_method, u_raw[fit_ix], abs_resid[fit_ix]
            )
            u_scaled_eval = apply_uncertainty_scaler(scaler_method, u_raw[eval_ix], params)
            score = score_uncertainty_candidate(
                u_scaled_eval, abs_resid[eval_ix], coverage, metric_weights
            )
        else:
            params = fit_uncertainty_scaler(scaler_method, u_raw, abs_resid)
            u_scaled_eval = apply_uncertainty_scaler(scaler_method, u_raw, params)
            score = score_uncertainty_candidate(
                u_scaled_eval, abs_resid, coverage, metric_weights
            )

        candidate_scores[cand] = score
        scaler_by_candidate[cand] = params

    if not candidate_scores:
        raise RuntimeError("Auto uncertainty scoring produced no valid candidates.")

    best_score = min(candidate_scores.values())
    tied = [c for c, s in candidate_scores.items() if abs(s - best_score) < 1e-12]
    selected = next(c for c in TIE_BREAK_ORDER if c in tied)

    # Refit scaler on full training OOF for deployment
    u_full = _raw_candidate_train(selected, Xy_data)
    assert u_full is not None
    final_params = fit_uncertainty_scaler(scaler_method, u_full, abs_resid)

    return {
        "selected": selected,
        "scaler_method": scaler_method,
        "scaler_params": final_params,
        "candidate_scores": candidate_scores,
        "coverage_target": coverage,
        "n_eval": int(n),
        "available_candidates": available,
    }


def apply_auto_uq(
    self: Any,
    Xy_data: Dict[str, Any],
    model_data: Mapping[str, Any],
    params_dir: str,
) -> Dict[str, Any]:
    """
    Select and attach auto-calibrated uncertainty columns to ``Xy_data``.

    Writes ``y_pred_{split}_uq_auto`` and run-level metadata under PREDICT/.
    """
    _ = params_dir
    if not bool(getattr(self.args, "uq_auto_enable", False)):
        return Xy_data

    if model_data["type"].lower() != "reg":
        clas_mode = str(getattr(self.args, "uq_auto_clas_mode", "error")).lower()
        if clas_mode == "error":
            raise ValueError(
                "uq_auto_enable is only supported for regression (problem_type='reg') "
                "in this release."
            )
        warnings.warn(
            "Auto uncertainty for classification is not calibrated; skipping.",
            UserWarning,
            stacklevel=2,
        )
        return Xy_data

    selection = evaluate_uq_candidates(Xy_data, self.args, model_data["type"])
    selected = selection["selected"]
    scaler_method = selection["scaler_method"]
    params = selection["scaler_params"]

    splits = ["train", "test"]
    if "X_external" in Xy_data:
        splits.append("external")

    for split in splits:
        u_raw = _raw_candidate_split(selected, split, Xy_data)
        if u_raw is None:
            continue
        u_auto = apply_uncertainty_scaler(scaler_method, u_raw, params)
        Xy_data[f"y_pred_{split}_uq_auto"] = np.asarray(u_auto, dtype=float).tolist()

    Xy_data["uq_auto_selected"] = selected
    Xy_data["uq_auto_scaler_params"] = params
    Xy_data["uq_auto_metadata"] = selection

    meta_path = Path("PREDICT") / "uq_auto_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: (v if not isinstance(v, dict) else dict(v))
        for k, v in selection.items()
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)

    return Xy_data


def _get_bo_ready_prediction_bundle(
    Xy_data: Mapping[str, Any],
    split: str,
    y_col: str,
) -> Dict[str, Any]:
    """
    Return mean, sigma, and provenance for Bayesian-optimization consumers.

    Uses auto uncertainty when present, otherwise CV SD.
    """
    pred_key = f"y_pred_{split}"
    auto_key = f"y_pred_{split}_uq_auto"
    sd_key = f"y_pred_{split}_sd"

    mean = np.asarray(Xy_data[pred_key], dtype=float)
    if auto_key in Xy_data:
        sigma = np.asarray(Xy_data[auto_key], dtype=float)
        source = str(Xy_data.get("uq_auto_selected", "auto"))
    elif sd_key in Xy_data:
        sigma = np.asarray(Xy_data[sd_key], dtype=float)
        source = "cv_sd_fallback"
    else:
        raise KeyError(f"No uncertainty available for split {split!r}.")

    return {
        "y_col": y_col,
        "mean": mean,
        "sigma": np.maximum(sigma, 0.0),
        "provenance": source,
    }
