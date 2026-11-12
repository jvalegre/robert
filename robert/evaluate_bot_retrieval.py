"""Measure robBOT evidence retrieval without provider calls.

Cases name acceptable evidence by stable entity ID or chunk ID. The report
includes ranked candidates so failures can be reviewed individually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from gui_easyrob.bot.bot_context import GuiSnapshot
from gui_easyrob.bot.bot_engine import BotEngine
from gui_easyrob.bot.question_intent import (
    classify_question_intent,
    is_generic_question,
    select_inference_history,
)


def _snapshot(case: Mapping[str, object]) -> GuiSnapshot:
    return GuiSnapshot(
        active_tab=str(case.get("active_tab", "")),
        workflow=str(case.get("workflow", "")),
        main_csv_path="",
        test_csv_path="",
        target_column="",
        prediction_type="",
        process_running=False,
        run_enabled=False,
        stop_enabled=False,
        recent_console="",
        console_terms=[],
    )


def _metrics(cases: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    count = len(cases)
    hits = sum(isinstance(case["rank"], int) and case["rank"] <= 3 for case in cases)
    first_hits = sum(case["rank"] == 1 for case in cases)
    reciprocal_sum = sum(
        1 / case["rank"] for case in cases if isinstance(case["rank"], int)
    )
    return {
        "count": count,
        "hits_at_3": hits,
        "hit_rate_at_1": first_hits / count if count else 0.0,
        "hit_rate_at_3": hits / count if count else 0.0,
        "mrr_at_5": reciprocal_sum / count if count else 0.0,
    }


def evaluate_cases(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Rank packaged evidence for each labeled question and follow-up."""
    engine = BotEngine()
    rows: list[dict[str, object]] = []
    for case in cases:
        question = str(case["question"])
        language = str(case["language"])
        expected = {str(value) for value in case["expected_evidence"]}
        history = case.get("history", [])
        selected_history = select_inference_history(question, history)
        intent = classify_question_intent(question, selected_history)
        generic = is_generic_question(question, intent)
        retrieval_question = engine._retrieval_question(question, selected_history)
        snapshot = _snapshot(case)
        candidates = engine._retrieve_candidates(retrieval_question, snapshot, intent, generic)
        rank = next(
            (
                index for index, candidate in enumerate(candidates, start=1)
                if expected & {candidate.chunk.id, candidate.chunk.entity_id}
            ),
            None,
        )
        rows.append({
            "id": str(case["id"]),
            "language": language,
            "intent": intent,
            "rank": rank,
            "matched_at_1": rank == 1,
            "matched_at_3": rank is not None and rank <= 3,
            "expected_evidence": sorted(expected),
            "top_evidence": [
                {
                    "id": candidate.chunk.id,
                    "entity_id": candidate.chunk.entity_id,
                    "title": candidate.chunk.title,
                    "score": round(candidate.score, 3),
                }
                for candidate in candidates
            ],
        })
    return {
        "summary": _metrics(rows),
        "by_language": {
            language: _metrics([row for row in rows if row["language"] == language])
            for language in sorted({row["language"] for row in rows})
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path,
        default=Path(__file__).parent / "tests/fixtures/bot_retrieval_cases.json",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    report = evaluate_cases(cases)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "by_language": report["by_language"]}, indent=2))
    for case in report["cases"]:
        if not case["matched_at_1"]:
            print(f"Miss: {case['id']} (rank={case['rank']})")


if __name__ == "__main__":
    main()
