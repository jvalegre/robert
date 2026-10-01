"""Conservative calibration of unsupported predictive-performance guarantees."""

import re
from typing import Sequence


_VALIDATION = re.compile(r"\b(?:validation|validación|validacion|CV)\b", re.I)
_PROMISE = r"(?:guarante\w*|ensur\w*|garantiz\w*|asegur\w*)"
_QUALITY = r"(?:reliab\w*|accura\w*|unseen|generaliz\w*|fiabl\w*|precis\w*|no vistos)"
_UNSUPPORTED = re.compile(rf"\b{_PROMISE}\b[^.!?\n]{{0,140}}\b{_QUALITY}\b", re.I)
_NEGATED = re.compile(rf"\b(?:no|not|never|cannot|can't|doesn't)\s+(?:\w+\s+){{0,2}}{_PROMISE}", re.I)


def calibrate_validation_guarantees(text: str) -> str:
    """Replace a narrow class of false guarantees, retaining correct caveats."""
    if not _VALIDATION.search(text):
        return text
    pieces = re.split(r"((?<=[.!?])\s+(?=[A-ZÁÉÍÓÚ])|\n+)", text)
    corrected = False
    spanish = False
    for index in range(0, len(pieces), 2):
        sentence = pieces[index]
        if sentence.rstrip().endswith("?") or _NEGATED.search(sentence):
            continue
        match = _UNSUPPORTED.search(sentence)
        if match:
            spanish = spanish or bool(re.search(r"garantiz|asegur", match.group(), re.I))
            pieces[index] = ""
            corrected = True
    if not corrected:
        return text
    caveat = (
        "La validación estima el rendimiento en datos no usados para entrenar; no garantiza predicciones fiables en datos nuevos."
        if spanish else
        "Validation estimates performance on data held out from training; it does not guarantee reliable predictions on new data."
    )
    remaining = re.sub(r"\n{3,}", "\n\n", "".join(pieces)).strip()
    return f"{remaining}\n\n{caveat}" if remaining else caveat


def result_summary_issues(
    text: str,
    *,
    metrics: Sequence[tuple[str, str, str, str]],
    unchanged_test_set_warning: bool,
    has_unresolved_verify_checks: bool = False,
) -> tuple[str, ...]:
    """Identify a few high-confidence contradictions before showing a report summary."""
    issues = []
    has_raw_test_rmse = any(
        "rmse" in label.casefold() and "test" in scope.casefold()
        for label, _, scope, _ in metrics
    )
    if has_raw_test_rmse and re.search(
        r"\b(?:raw|unit[- ]based|original[- ]unit)\b[^.!?\n]{0,70}\bRMSE\b[^.!?\n]{0,90}"
        r"\b(?:not (?:listed|provided|available|reported)|missing|absent|no values)\b|"
        r"\b(?:no|without)\b[^.!?\n]{0,55}\b(?:raw|unit[- ]based)\b[^.!?\n]{0,50}\bRMSE\b",
        text, re.I,
    ):
        issues.append("Raw test RMSE is present in PREDICT metrics; report its value rather than saying it is absent.")
    if re.search(
        r"\bone[- ‑]?hot\b[^.!?\n]{0,240}\bcategorical descriptors?\b|"
        r"\bcategorical descriptors?\b[^.!?\n]{0,240}\bone[- ‑]?hot\b|"
        r"\bzero[- /‑]?non[- ‑]?zero\b[^.!?\n]{0,100}\bcategorical descriptors?\b",
        text, re.I,
    ) and not re.search(r"\bnot (?:a |about )?categorical descriptor", text, re.I):
        issues.append("VERIFY onehot tests zero/nonzero descriptor indicators; it does not identify categorical descriptors.")
    if unchanged_test_set_warning and re.search(
        r"\b(?:test[- ‑ ]set|test_set)\b[^.!?\n]{0,100}\b(?:forced|raised|changed|increased)\b|"
        r"\b(?:forced|raised|changed|increased)\b[^.!?\n]{0,100}\b(?:test[- ‑ ]set|test_set)\b",
        text, re.I,
    ):
        issues.append("The GENERATE warning reports 0.2 before and after; do not claim the test-set size changed.")
    if re.search(
        r"\b(?:scores?|puntuaci[oó]n(?:es)?)\b[^.!?\n]{0,95}\b(?:moderate|modest|strong|good|poor|bad|"
        r"moderad[ao]|buen[ao]|mal[ao])\b[^.!?\n]{0,35}\b(?:performance|rendimiento|model|modelo)\b",
        text, re.I,
    ):
        issues.append("A ROBERT score is a checklist summary, not an absolute label for predictive quality.")
    if re.search(
        r"\b(?:pass(?:ed|ing)?|superad[oa]s?)\b[^.!?\n]{0,100}\b(?:not|no)\s+(?:simply\s+)?"
        r"(?:memori[sz]\w*|overfit\w*|sobreajust\w*)\b",
        text, re.I,
    ):
        issues.append("Passing baseline checks does not prove the model has not memorized or overfit.")
    if re.search(
        r"\b(?:feature engineering|stratified sampling|re-encod\w*|reencod\w*|"
        r"collect(?:ing)? more data|recog(?:er|ida) m[aá]s datos|muestreo estratificado)\b",
        text, re.I,
    ):
        issues.append("Suggest inspecting recorded report checks and CSVs, not unverified model redesign or data collection.")
    if has_unresolved_verify_checks and re.search(
        r"\b(?:all|every)\b[^.!?\n]{0,45}\bVERIFY\b[^.!?\n]{0,30}\bpass(?:ed|es)?\b|"
        r"\bpass(?:ed|es)?\b[^.!?\n]{0,30}\b(?:all|every)\b[^.!?\n]{0,45}\bVERIFY\b",
        text, re.I,
    ):
        issues.append("At least one VERIFY check is FAILED or UNCLEAR; not all checks passed.")
    if re.search(
        r"\b(?:more|most|highly|very)\s+(?:reliable|robust|stable|fiable|robusto|estable)\b|"
        r"\b(?:stronger|better)\s+(?:model\s+)?stability\b",
        text, re.I,
    ):
        issues.append("Reliability or model stability cannot be inferred from the score or R2 difference alone.")
    return tuple(issues)
