"""Evaluate real workflow answers using a saved or transient Groq key.

Provider calls are billed to the configured account. Credentials are never exported.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import sys
import getpass

from PySide6.QtCore import QSettings
from gui_easyrob.bot.bot_context import GuiSnapshot
from gui_easyrob.bot.bot_engine import BotEngine
from gui_easyrob.bot.workflow_results import WorkflowResultStore, render_workflow_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=20)
    parser.add_argument("--key-stdin", action="store_true", help="Read a transient key from standard input.")
    parser.add_argument("--only", type=int, help="Run one numbered evaluation case.")
    parser.add_argument("--diagnose", action="store_true", help="Print provider error metadata without credentials.")
    args = parser.parse_args()
    settings = QSettings("easyROB", "easyROB")
    key = (
        (getpass.getpass("Groq API key (not saved): ") if sys.stdin.isatty() else sys.stdin.readline()).strip()
        if args.key_stdin else str(settings.value("bot/api_key", ""))
    )
    if not key or (not args.key_stdin and str(settings.value("bot/provider", "Groq")) != "Groq"):
        raise SystemExit("Save a Groq key in EasyROB Settings before this evaluation.")
    cases = (
        ("robert", "Summarize the ROBERT report for a beginner. Explain what needs attention, why it matters and what to check next."),
        ("robert", "Did all VERIFY checks pass? Can I trust this model?"),
        ("robert", "Resume el informe de ROBERT para una persona sin experiencia. Explica los resultados y donde poner atencion."),
        ("aqme", "Summarize this workflow for a beginner. What was generated and are there molecule failures?"),
        ("aqme", "What is the test R2 of this workflow?"),
        ("robert", "Which variant should I examine first, PFI or No PFI, and why?"),
    )
    roots = {
        "robert": Path("benchmark_results/workflow_results_real_20260917").resolve(),
        "aqme": Path("benchmark_results/workflow_results_aqme_real_20260917").resolve(),
    }
    engine, store, history, rows = BotEngine(), WorkflowResultStore(), [], []
    if args.diagnose:
        adapter = engine._provider_registry["Groq"]
        original_generate = adapter.generate_answer_with_usage

        def diagnosed_generate(**kwargs):
            try:
                return original_generate(**kwargs)
            except Exception as exc:
                print(
                    f"Provider error: {type(exc).__name__}, "
                    f"kind={getattr(exc, 'kind', None)}, status={getattr(exc, 'status_code', None)}",
                    file=sys.stderr, flush=True,
                )
                raise

        adapter.generate_answer_with_usage = diagnosed_generate
    args.output.mkdir(parents=True, exist_ok=True)
    for index, (kind, question) in enumerate(cases):
        if args.only is not None and args.only != index + 1:
            continue
        if index:
            time.sleep(max(0, args.interval))
        result = store.refresh(str(roots[kind]), "")
        snapshot = GuiSnapshot("ROBERT", "Full Workflow", "input.csv", "", "", "regression",
                               False, True, False, "", [], workflow_results=result)
        started = time.monotonic()
        answer = engine.answer_with_metadata(question, snapshot, "Cloud AI", "Groq", key,
                                             conversation_history=history, web_search_enabled=False)
        text = answer.text.replace(key, "[redacted]")
        rows.append(dict(case=index + 1, workflow=kind, question=question, answer=text,
                         provider_answer=answer.usage.total_tokens > 0,
                         seconds=round(time.monotonic()-started, 2), usage=asdict(answer.usage),
                         evidence=render_workflow_evidence(result, question, full=True, max_chars=16000)))
        history.extend([dict(role="user", content=question), dict(role="assistant", content=text)])
        (args.output / "answers.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Case {index+1}: {kind}, {len(text.split())} words, {answer.usage.total_tokens} tokens", flush=True)


if __name__ == "__main__":
    main()
