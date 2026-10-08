"""
Evaluate simulated interview results with the InterviewReportCard suite,
via the Gemini Batch API — mirrors human_data/run_human_evaluation.py so the two
are directly comparable, but reads Study.save() result files (produced by
run_interviews.py) instead of raw human session exports.

For each (preset, agent) combo, submits the three LLM-judged evaluation
dimensions (participant responses, interviewer behavior, participant experience) as
Gemini batch jobs over that combo's transcripts, then polls until they
complete and writes a combined result file with both per-participant values
(tagged by participant name) and the aggregate across all participants.
conversation_length (avg_turns, avg_response_length) is added at retrieve
time straight from the transcripts — no LLM calls, no batch job.

Resumable: submission skips any combo already recorded in eval_handles.json,
and retrieval skips any combo whose output file already exists.

Usage:
    python run_evaluation.py                       # submit + poll + retrieve
    python run_evaluation.py --submit-only
    python run_evaluation.py --retrieve-only        # resume
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from datetime import datetime, timezone

import experiment_config as cfg

from interviewplayground import Study
from interviewplayground.interviewreportcard import (
    submit_participant_responses, retrieve_participant_responses,
    submit_interviewer_behavior, retrieve_interviewer_behavior,
    submit_participant_experience, retrieve_participant_experience,
    evaluate_conversation_length,
)
from interviewplayground.llm_client import LLMClient

BASE_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
EVAL_DIR = BASE_DIR / "evaluations"
HANDLES_PATH = BASE_DIR / "eval_handles.json"

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _batch_ids(entry: dict) -> list[str]:
    ids = []
    for dim in ("participant_responses", "interviewer_behavior", "participant_experience"):
        h = entry.get(dim) or {}
        for k, v in h.items():
            if k.endswith("batch_id") and v:
                ids.append(v)
    return ids


def _load_handles() -> dict:
    if HANDLES_PATH.exists():
        return json.loads(HANDLES_PATH.read_text())
    return {}


def _save_handles(handles: dict) -> None:
    HANDLES_PATH.write_text(json.dumps(handles, indent=2, ensure_ascii=False))


def combo_key(preset: str, agent_id: str) -> str:
    return f"{preset}__{agent_id}"


def submit_all(presets: list[str], agents: list[str]) -> dict:
    cfg.load_env()
    handles = _load_handles()

    for preset in presets:
        interview_guide = cfg.get_study_config(preset)["interview_spec"]
        for agent_id in agents:
            key = combo_key(preset, agent_id)
            if key in handles:
                log(f"skip submit (already submitted) {key}")
                continue

            result_path = RESULTS_DIR / f"{key}.json"
            if not result_path.exists():
                log(f"skip submit (no result file) {key}")
                continue

            study = Study.load(str(result_path))
            transcripts = [p.transcript for p in study.participants]
            names = [p.name or f"participant_{i}" for i, p in enumerate(study.participants)]
            entry = {
                "preset": preset, "agent": agent_id, "study_id": cfg.PRESET_TO_STUDY[preset],
                "names": names, "model": cfg.JUDGE_MODEL,
            }
            try:
                entry["participant_responses"] = submit_participant_responses(
                    transcripts, study.research_questions, interview_guide,
                    model=cfg.JUDGE_MODEL,
                )
                entry["interviewer_behavior"] = submit_interviewer_behavior(
                    transcripts, model=cfg.JUDGE_MODEL,
                )
                entry["participant_experience"] = submit_participant_experience(
                    transcripts, model=cfg.JUDGE_MODEL,
                )
                log(f"submitted {key} ({len(names)} participants)")
            except Exception as exc:
                entry["submit_error"] = str(exc)
                log(f"submit FAILED {key}: {exc}")

            handles[key] = entry
            _save_handles(handles)

    return handles


def poll_evals(handles: dict, poll_interval: int, max_wait: int) -> dict:
    client = LLMClient(model=cfg.JUDGE_MODEL)
    pending = {}
    for key, entry in handles.items():
        if "submit_error" in entry:
            continue
        if (EVAL_DIR / f"{key}.json").exists():
            continue
        for bid in _batch_ids(entry):
            pending[bid] = key

    log(f"polling {len(pending)} batch job(s) across {len(set(pending.values()))} combo(s)")
    final: dict[str, str] = {}
    start = time.monotonic()
    while pending:
        time.sleep(poll_interval)
        for bid in list(pending):
            try:
                st = client.batch_status(bid)
            except Exception as exc:
                log(f"  status error {bid[-10:]}: {exc}")
                continue
            if st in TERMINAL_STATUSES:
                final[bid] = st
                log(f"  {pending[bid]} {bid[-10:]}: {st} ({len(final)} done, {len(pending) - 1} left)")
                del pending[bid]
        if time.monotonic() - start > max_wait:
            log(f"  TIMEOUT: {len(pending)} batch(es) still pending")
            break
    return final


def _tag_per_participant(result: dict, names: list[str]) -> dict:
    tagged = [
        {"name": n, **pp}
        for n, pp in zip(names, result.get("per_participant", []))
    ]
    return {**result, "per_participant": tagged}


def retrieve_evals(handles: dict, batch_status: dict) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for key, entry in handles.items():
        out_path = EVAL_DIR / f"{key}.json"
        if out_path.exists():
            continue
        if "submit_error" in entry:
            log(f"  retrieve skip {key} (submit_error)")
            continue
        ids = _batch_ids(entry)
        if any(batch_status.get(bid) != "completed" for bid in ids):
            log(f"  retrieve skip {key} (batch not all completed)")
            continue

        try:
            results = {
                "participant_responses": retrieve_participant_responses(entry["participant_responses"]),
                "interviewer_behavior": retrieve_interviewer_behavior(entry["interviewer_behavior"]),
                "participant_experience": retrieve_participant_experience(entry["participant_experience"]),
            }
        except Exception as exc:
            log(f"  retrieve FAILED {key}: {exc}")
            continue

        names = entry["names"]
        results = {dim: _tag_per_participant(r, names) for dim, r in results.items()}

        result_path = RESULTS_DIR / f"{key}.json"
        study = Study.load(str(result_path))
        transcripts = [p.transcript for p in study.participants]
        cl = evaluate_conversation_length(transcripts)
        results["conversation_length"] = {
            **cl,
            "per_participant": [
                {"name": n, **p} for n, p in zip(names, cl["per_participant"])
            ],
        }

        payload = {
            "preset": entry["preset"], "agent": entry["agent"], "study_id": entry["study_id"],
            "judge_model": entry["model"], "n_participants": len(names), "names": names,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        log(f"  wrote eval -> {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="obesity_weight_management", choices=cfg.PRESETS + ["all"])
    parser.add_argument("--agent", default="all", choices=cfg.AGENTS + ["all"])
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--retrieve-only", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--max-wait", type=int, default=21600)
    args = parser.parse_args()

    cfg.load_env()

    presets = cfg.PRESETS if args.preset == "all" else [args.preset]
    agents = cfg.AGENTS if args.agent == "all" else [args.agent]

    if args.retrieve_only:
        handles = _load_handles()
    else:
        handles = submit_all(presets, agents)
        if args.submit_only:
            return

    batch_status = poll_evals(handles, args.poll_interval, args.max_wait)
    retrieve_evals(handles, batch_status)


if __name__ == "__main__":
    main()
