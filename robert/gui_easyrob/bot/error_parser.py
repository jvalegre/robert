from __future__ import annotations

from dataclasses import dataclass
import re

__all__ = ["ParsedFailure", "infer_target_column_failure_hint", "parse_python_failure"]


_TRACEBACK_RE = re.compile(r'^File "([^"]+)", line (\d+), in (.+)$')
_EXCEPTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):\s*(.+)$")


@dataclass(frozen=True, slots=True)
class ParsedFailure:
    failure_type: str = ""
    failure_message: str = ""
    failure_location: str = ""
    failure_function: str = ""
    failure_operation: str = ""
    likely_cause: str = ""


def _likely_cause(failure_type: str, failure_message: str, failure_operation: str) -> str:
    lowered_message = failure_message.lower()
    lowered_operation = failure_operation.lower()
    if failure_type == "TypeError" and "not supported for the input types" in lowered_message:
        if "linregress" in lowered_operation:
            return "one or both inputs to the regression step are not numeric or have incompatible dtypes"
        return "the operation received values with incompatible Python or NumPy types"
    return ""


def infer_target_column_failure_hint(
    target_column: str,
    prediction_type: str,
    failure_type: str,
    failure_operation: str,
    likely_cause: str,
) -> str:
    target = str(target_column or "").strip()
    prediction = str(prediction_type or "").strip().lower()
    operation = str(failure_operation or "").lower()
    cause = str(likely_cause or "").lower()
    if not target:
        return ""
    if failure_type != "TypeError":
        return ""
    if "linregress" not in operation:
        return ""
    if "not numeric" not in cause and "incompatible dtypes" not in cause:
        return ""

    target_lower = target.lower()
    if target_lower == "smiles" or "smiles" in target_lower:
        return (
            f"The selected target column is `{target}`, which is text structure data rather than a numeric response. "
            f"For {prediction or 'this'} modelling, the target should be the property you want to predict, not the SMILES column."
        )
    if target_lower in {"code_name", "name"} or "name" in target_lower:
        return (
            f"The selected target column is `{target}`, which looks like an identifier/text column rather than a numeric response. "
            f"For {prediction or 'this'} modelling, the target should be the property you want to predict."
        )
    return ""


def parse_python_failure(console_text: str) -> ParsedFailure:
    lines = [line.strip() for line in str(console_text or "").splitlines() if line.strip()]
    if not lines:
        return ParsedFailure()

    failure_type = ""
    failure_message = ""
    failure_location = ""
    failure_function = ""
    failure_operation = ""

    for line in reversed(lines):
        exception_match = _EXCEPTION_RE.match(line)
        if exception_match:
            failure_type = exception_match.group(1).strip()
            failure_message = exception_match.group(2).strip()
            break

    for index, line in enumerate(lines):
        traceback_match = _TRACEBACK_RE.match(line)
        if traceback_match:
            file_name = traceback_match.group(1).strip()
            line_number = traceback_match.group(2).strip()
            function_name = traceback_match.group(3).strip()
            failure_location = f"{file_name}:{line_number}"
            failure_function = function_name
            if index + 1 < len(lines):
                next_line = lines[index + 1].strip()
                if next_line and not next_line.startswith("^") and not next_line.startswith('File "'):
                    failure_operation = next_line

    return ParsedFailure(
        failure_type=failure_type,
        failure_message=failure_message,
        failure_location=failure_location,
        failure_function=failure_function,
        failure_operation=failure_operation,
        likely_cause=_likely_cause(failure_type, failure_message, failure_operation),
    )
