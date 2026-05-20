"""
Platform-specific integration-test goldens (Linux vs Windows CI).

Values are ROBERT log metrics formatted with :.2. Re-baseline both tables when
sklearn/BO stack changes (.circleci/config.yml conda env).
"""

from __future__ import annotations

import math
import re
import sys

# Tight tolerance: guards parse noise only, not cross-platform drift (separate goldens).
_METRIC_REL_TOL = 0.01
_METRIC_ABS_TOL = 0.005


def platform_key() -> str:
    """``linux`` or ``win32`` (CircleCI Windows uses win32)."""
    return "win32" if sys.platform == "win32" else "linux"


def log_line_metric_close(
    line: str,
    prefix: str,
    expected: float,
    *,
    rel_tol: float = _METRIC_REL_TOL,
    abs_tol: float = _METRIC_ABS_TOL,
) -> bool:
    """True if the first numeric token after ``prefix`` matches ``expected``."""
    idx = line.find(prefix)
    if idx == -1:
        return False
    rest = line[idx + len(prefix) :].lstrip()
    m = re.match(r"([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)", rest)
    if not m:
        return False
    return math.isclose(float(m.group(1)), expected, rel_tol=rel_tol, abs_tol=abs_tol)


# VERIFY standard regression (PFI model section): (CV RMSE, +15% threshold, +30% threshold)
VERIFY_STANDARD_RMSE = {
    "linux": (0.27, 0.31, 0.35),
    "win32": (0.27, 0.31, 0.35),
}

# VERIFY standard regression: y_shuffle flawed-model RMSE (:.2 in log)
VERIFY_STANDARD_Y_SHUFFLE_RMSE = {
    "linux": 0.94,
    "win32": 0.97,
}

# GENERATE standard job: log prefix -> expected combined RMSE
GENERATE_STANDARD_RMSE = {
    "linux": {
        "o Best combined RMSE (target) found in BO for RF (no PFI filter):": 0.64,
        "o Combined RMSE for RF (with PFI filter):": 0.77,
        "o Best combined RMSE (target) found in BO for GB (no PFI filter):": 0.49,
        "o Combined RMSE for GB (with PFI filter):": 0.41,
        "o Best combined RMSE (target) found in BO for NN (no PFI filter):": 0.37,
        "o Combined RMSE for NN (with PFI filter):": 0.39,
        "o Combined RMSE for MVL (no BO needed) (no PFI filter):": 0.51,
        "o Combined RMSE for MVL (with PFI filter):": 0.44,
    },
    "win32": {
        "o Best combined RMSE (target) found in BO for RF (no PFI filter):": 0.62,
        "o Combined RMSE for RF (with PFI filter):": 0.75,
        "o Best combined RMSE (target) found in BO for GB (no PFI filter):": 0.45,
        "o Combined RMSE for GB (with PFI filter):": 0.41,
        "o Best combined RMSE (target) found in BO for NN (no PFI filter):": 0.36,
        "o Combined RMSE for NN (with PFI filter):": 0.37,
        "o Combined RMSE for MVL (no BO needed) (no PFI filter):": 0.47,
        "o Combined RMSE for MVL (with PFI filter):": 0.47,
    },
}


def assert_verify_standard_y_shuffle_line(line: str) -> None:
    """Assert the y_shuffle flawed-model line for standard VERIFY tests."""
    expected = VERIFY_STANDARD_Y_SHUFFLE_RMSE[platform_key()]
    assert "o y_shuffle: PASSED" in line
    assert f"RMSE = {expected:.2}" in line, (
        f"y_shuffle RMSE line {line!r} vs golden {expected:.2} ({platform_key()})"
    )


def assert_verify_standard_rmse_line(line: str) -> None:
    """Assert the Original RMSE summary line for standard VERIFY tests."""
    assert "Original RMSE (10x 5-fold CV)" in line
    cv_rmse, t15, t30 = VERIFY_STANDARD_RMSE[platform_key()]
    pattern = (
        r"Original RMSE \(10x 5-fold CV\)\s+"
        r"([-+]?(?:\d*\.\d+|\d+))\s+\+\s+15%\s+&\s+30%\s+threshold\s+=\s+"
        r"([-+]?(?:\d*\.\d+|\d+))\s+&\s+([-+]?(?:\d*\.\d+|\d+))"
    )
    m = re.search(pattern, line)
    assert m, f"Could not parse VERIFY RMSE line: {line!r}"
    actual_cv, actual_t15, actual_t30 = (float(m.group(i)) for i in range(1, 4))
    assert math.isclose(
        actual_cv, cv_rmse, rel_tol=_METRIC_REL_TOL, abs_tol=_METRIC_ABS_TOL
    ), f"CV RMSE {actual_cv} vs golden {cv_rmse} ({platform_key()})"
    assert math.isclose(
        actual_t15, t15, rel_tol=_METRIC_REL_TOL, abs_tol=_METRIC_ABS_TOL
    ), f"15% threshold {actual_t15} vs golden {t15} ({platform_key()})"
    assert math.isclose(
        actual_t30, t30, rel_tol=_METRIC_REL_TOL, abs_tol=_METRIC_ABS_TOL
    ), f"30% threshold {actual_t30} vs golden {t30} ({platform_key()})"


def count_generate_standard_rmse_matches(outlines: list[str]) -> int:
    """Count how many standard-job BO RMSE log lines match platform goldens."""
    goldens = GENERATE_STANDARD_RMSE[platform_key()]
    count = 0
    for line in outlines:
        for prefix, expected in goldens.items():
            if log_line_metric_close(line, prefix, expected):
                count += 1
                break
    return count


def assert_generate_standard_rmse_matches(outlines: list[str]) -> None:
    """Assert all eight standard-job BO RMSE log lines match platform goldens."""
    goldens = GENERATE_STANDARD_RMSE[platform_key()]
    expected_count = len(goldens)
    actual_count = count_generate_standard_rmse_matches(outlines)
    if actual_count == expected_count:
        return
    mismatches = []
    for prefix, golden in goldens.items():
        for line in outlines:
            if prefix not in line:
                continue
            idx = line.find(prefix)
            rest = line[idx + len(prefix) :].lstrip()
            m = re.match(r"([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)", rest)
            if not m:
                mismatches.append(f"{prefix}: could not parse {line!r}")
                break
            actual = float(m.group(1))
            if not log_line_metric_close(line, prefix, golden):
                mismatches.append(f"{prefix}: log={actual:.2f} golden={golden:.2f}")
            break
        else:
            mismatches.append(f"{prefix}: line not found in GENERATE_data.dat")
    raise AssertionError(
        f"GENERATE standard RMSE matches {actual_count}/{expected_count} "
        f"({platform_key()}): " + "; ".join(mismatches)
    )
