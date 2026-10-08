"""
Run preset participant-simulation studies against interviewer agents, for the
predictive-validity comparison against the real human study.

Loads a preset Study, has the chosen interviewer agent interview each participant
turn-by-turn (agent.generate_question <-> participant.ask), and saves the
completed Study (with transcripts) via Study.save(). Participants are interviewed
in parallel across a thread pool.

Each interview runs until an estimated elapsed-time cap (--time-limit-minutes,
default 30 — matching the human study): interviewer turns count real generation
wall-clock, participant turns count estimated speaking time
(interviewplayground.speaking_time). Per-participant timing is recorded in the
run's .meta.json.

Interviewer agents keep their built-in DEFAULT_MODEL (gpt-5.4-mini, same as the
human study); only the participant simulator's model is configurable here
(default gemini/gemini-3.1-pro-preview).

Checkpointed + resumable: each combo's result file is re-saved after every
completed participant (not just at the very end), and re-running skips any
participant that already has a non-empty transcript in an existing result
file — so a killed job (time limit, crash) only loses whatever was in flight,
and a re-run picks up where it left off.

Usage:
    # obesity study, all 5 interviewers, 150 interviews total
    python run_interviews.py --preset obesity_weight_management --agent all

    # single combo, sequential (debugging)
    python run_interviews.py --preset obesity_weight_management --agent storysage --workers 1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from time import perf_counter

import experiment_config as cfg
from interviewplayground.speaking_time import estimate_speaking_duration

RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
_print_lock = threading.Lock()


def _log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def _make_agent(agent_class, study_id: str, study_cfg: dict):
    """Instantiate a fresh interviewer agent for one participant (own session)."""
    return agent_class(
        pre_survey_data={},
        study_id=study_id,
        session_id=str(uuid.uuid4()),
        interview_description=study_cfg.get("interview_description", ""),
        interview_spec=study_cfg.get("interview_spec"),
    )


def interview(agent, participant, time_limit_s: float, max_turns: int) -> dict:
    """Drive one interview until the estimated elapsed interview time reaches
    time_limit_s (or the agent ends, or the max_turns safety backstop is hit).

    Each turn's elapsed time is the interviewer's real generation wall-clock
    (perf_counter around generate_question) plus the participant's estimated
    speaking time (estimate_speaking_duration of the answer). The turn that
    crosses time_limit_s is kept, then the interview stops.

    Sets agent.elapsed_seconds before each turn, mirroring what the production
    backend's researcher/generate_question.py does before calling the agent
    directly (which this harness otherwise bypasses entirely). Without it,
    every agent's built-in end_allowed() (_end_interview.py) permanently reads
    elapsed_seconds as its 0.0 default and never offers the model the
    end_interview option — so the interview can only ever stop via the
    external time_limit_s/max_turns caps here, not the agent's own judgment
    that the guide is sufficiently covered.
    """
    full_transcript: list[dict] = []
    current_turn = ""
    turn_number = 1
    total_seconds = 0.0
    turns: list[dict] = []
    stopped_reason = "max_turns"
    while turn_number <= max_turns:
        agent.elapsed_seconds = total_seconds
        t0 = perf_counter()
        question = agent.generate_question(turn_number, current_turn, full_transcript)
        interviewer_seconds = perf_counter() - t0
        if not question:
            # None = explicit end signal. An empty string shows up occasionally
            # (e.g. mimitalk emitting {"question_to_ask": ""} on a malformed
            # generation) — feeding that to participant.ask() crashes memory
            # retrieval (OpenAI's embeddings API rejects empty input), so treat
            # it the same as an end signal rather than erroring the interview out.
            stopped_reason = "agent_end"
            break
        answer = participant.ask(question)
        participant_seconds = estimate_speaking_duration(answer, participant.verbosity)
        total_seconds += interviewer_seconds + participant_seconds
        full_transcript.append({
            "turn_number": turn_number,
            "question_text": question,
            "transcript": answer,
        })
        turns.append({
            "turn": turn_number,
            "interviewer_seconds": round(interviewer_seconds, 3),
            "participant_seconds": round(participant_seconds, 3),
            "cumulative_seconds": round(total_seconds, 3),
        })
        current_turn = answer
        if total_seconds >= time_limit_s:
            stopped_reason = "time_limit"
            break
        turn_number += 1
    return {
        "total_seconds": round(total_seconds, 3),
        "n_turns": len(turns),
        "stopped_reason": stopped_reason,
        "turns": turns,
    }


def _write_meta(out_path: pathlib.Path, preset: str, agent_id: str, study_id: str,
                n_participants: int, time_limit_minutes: float, max_turns: int,
                model: str | None, timings_by_pid: dict, study) -> None:
    participant_timings = [
        {"name": p.name or "unnamed", **timings_by_pid.get(id(p), {})}
        for p in study.participants
    ]
    meta = {
        "preset": preset,
        "agent": agent_id,
        "study_id": study_id,
        "n_participants": n_participants,
        "time_limit_minutes": time_limit_minutes,
        "max_turns": max_turns,
        "participant_model": model or cfg.PARTICIPANT_MODEL,
        "interviewer_model": "gpt-5.4-mini (agent default)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "participants": participant_timings,
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="obesity_weight_management",
                        choices=cfg.PRESETS + ["all"])
    parser.add_argument("--agent", default="all", choices=cfg.AGENTS + ["all"])
    parser.add_argument("--time-limit-minutes", type=float, default=30.0,
                        help="Per-participant interview time cap in minutes (default: 30, "
                             "matching the human study).")
    parser.add_argument("--max-turns", type=int, default=200,
                        help="Safety backstop on turns (default: 200); the time limit is the "
                             "primary cap.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel interview workers (default: 8; use 1 for sequential).")
    parser.add_argument("--model", default=None,
                        help=f"Participant completion model (default: {cfg.PARTICIPANT_MODEL}).")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    cfg.load_env()

    from interviewplayground import set_default_model, load_preset

    model = args.model or cfg.PARTICIPANT_MODEL
    set_default_model(model)

    presets = cfg.PRESETS if args.preset == "all" else [args.preset]
    agents = cfg.AGENTS if args.agent == "all" else [args.agent]
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from interviewplayground import Study

    combo_studies: dict[tuple[str, str], object] = {}
    combo_paths: dict[tuple[str, str], pathlib.Path] = {}
    tasks = []
    n_skipped = 0
    for preset in presets:
        study_cfg = cfg.get_study_config(preset)
        study_id = cfg.PRESET_TO_STUDY[preset]
        for agent_id in agents:
            agent_class = cfg.get_interviewer_class(agent_id)
            out_path = output_dir / f"{preset}__{agent_id}.json"
            if out_path.exists():
                study = Study.load(str(out_path))
                _log(f"resuming {preset}/{agent_id} from existing {out_path.name}")
            else:
                study = load_preset(preset)
            combo_studies[(preset, agent_id)] = study
            combo_paths[(preset, agent_id)] = out_path
            for participant in study.participants:
                if participant.transcript:
                    n_skipped += 1
                    continue
                agent = _make_agent(agent_class, study_id, study_cfg)
                tasks.append(((preset, agent_id), participant, agent))

    time_limit_s = args.time_limit_minutes * 60
    _log(f"Running {len(combo_studies)} study/agent combos "
         f"({len(tasks)} participant interviews, {n_skipped} already done) with "
         f"{args.workers} worker(s), time_limit={args.time_limit_minutes:g} min "
         f"(max_turns backstop={args.max_turns}).")

    failures: list[tuple[tuple[str, str], str, str]] = []
    timings_by_pid: dict[int, dict] = {}
    save_lock = threading.Lock()

    def _checkpoint(combo: tuple[str, str]) -> None:
        with save_lock:
            study = combo_studies[combo]
            out_path = combo_paths[combo]
            study.save(str(out_path))
            _write_meta(out_path, combo[0], combo[1], cfg.PRESET_TO_STUDY[combo[0]],
                        study.n, args.time_limit_minutes, args.max_turns, model,
                        timings_by_pid, study)

    def _run(combo, participant, agent):
        timing = interview(agent, participant, time_limit_s, args.max_turns)
        return combo, participant, timing

    if not tasks:
        _log("nothing to do — every participant already has a transcript.")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run, c, p, a): (c, p) for (c, p, a) in tasks}
        done = 0
        for future in as_completed(futures):
            combo, participant = futures[future]
            done += 1
            name = participant.name or "unnamed"
            try:
                _, _, timing = future.result()
                timings_by_pid[id(participant)] = timing
                _log(f"[{done}/{len(tasks)}] {combo[0]}/{combo[1]} :: {name} — "
                     f"{timing['n_turns']} turns, ~{timing['total_seconds'] / 60:.1f} min "
                     f"({timing['stopped_reason']})")
            except Exception as exc:
                failures.append((combo, name, str(exc)))
                _log(f"[{done}/{len(tasks)}] {combo[0]}/{combo[1]} :: {name} — FAILED: {exc}")
                # A crash can leave a partial transcript (e.g. a dangling interviewer
                # turn with no reply); clear it so this participant isn't mistaken
                # for "done" on resume, and gets a clean full retry instead.
                participant.transcript = []
            finally:
                # Checkpoint after every completed participant (success or failure)
                # so a killed job never loses more than the interviews still in flight.
                _checkpoint(combo)

    if failures:
        _log(f"\n{len(failures)} interview(s) failed:")
        for combo, name, err in failures:
            _log(f"  {combo[0]}/{combo[1]} :: {name}: {err}")


if __name__ == "__main__":
    main()
