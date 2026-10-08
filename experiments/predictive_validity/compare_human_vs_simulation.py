"""
Compare human vs. simulated interviews for one preset, per metric, across the
5 interviewers (5 paired data points per metric: one human aggregate and one
simulation aggregate per interviewer).

Most metrics are read straight from each evaluation file's top-level aggregate
— interviewplayground.interviewreportcard's own per-participant-average formulas
(see src/interviewplayground/interviewreportcard/), including conversation_length
(avg_turns, avg_response_length), which the package computes directly from
transcripts with no LLM calls.

comfort_level and overall_experience are the exception: on the human side,
these come from participants' own post-survey self-reports (post_survey_data
in the session export) rather than the LLM judge's evaluation of the human
transcripts — self-report is the ground truth these two metrics are meant to
approximate, so the simulation's LLM-judged comfort/experience is compared
directly against it instead of against another LLM judgment. Every other
metric still uses the human LLM judge (self-report has no equivalent for
coverage/behavior metrics). The post-survey's "comfort"/"rating" questions use
the exact same rubric wording as the judge prompts (see
human_data/compare_self_report.py), so they convert to the same 1-4/1-5 scales
with no rescaling needed.

Human participant counts per interviewer needn't match the simulation's 30 —
the comparison is over each interviewer's aggregate, not paired participants
(relevant for asian_american_politics, whose human data collection is still
short of 30 for 4 of its 5 interviewers).

Usage:
    python compare_human_vs_simulation.py                                   # obesity_weight_management
    python compare_human_vs_simulation.py --preset asian_american_politics
    python compare_human_vs_simulation.py --preset genai_knowledge_work
"""
from __future__ import annotations

import argparse
import json
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
HUMAN_DATA_DIR = REPO_ROOT / "human_data"
HUMAN_EVAL_DIR = HUMAN_DATA_DIR / "evaluations"
SIM_EVAL_DIR = BASE_DIR / "evaluations"
OUT_DIR = BASE_DIR / "comparisons"

AGENTS = ["interviewgpt", "llmroleplay", "mimitalk", "sparkme", "storysage"]

METRICS = [
    "avg_turns", "avg_response_length",
    "relevant_response_volume_avg",
    "interview_guide_coverage",
    "novel_responses_avg",
    "coherence", "adaptiveness",
    "leading_questions_avg", "support_rapport_avg", "unclear_questions_avg",
    "comfort_level", "overall_experience",
]

# comfort_level/overall_experience use self-report as the human ground truth
# (see module docstring) instead of the LLM judge's evaluation of human
# transcripts, which every other metric still uses.
SELF_REPORT_METRICS = {"comfort_level", "overall_experience"}

COMFORT_MAP = {
    "Entirely uncomfortable or treated unfairly": 1,
    "Mostly uncomfortable, but with some exceptions": 2,
    "Mostly comfortable, but with some exceptions": 3,
    "Entirely comfortable and treated fairly": 4,
}
RATING_MAP = {
    "Poor": 1,
    "Below Average": 2,
    "Average": 3,
    "Good": 4,
    "Excellent": 5,
}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r
    return _pearson(ranks(xs), ranks(ys))


def load_eval(directory: pathlib.Path, preset: str, agent: str) -> dict:
    return json.loads((directory / f"{preset}__{agent}.json").read_text())


def self_report_by_agent(preset: str) -> dict[str, dict]:
    """Mean self-reported comfort_level/overall_experience per interviewer,
    straight from post-survey responses (independent of any LLM judge)."""
    payload = json.loads((HUMAN_DATA_DIR / f"{preset}_sessions.json").read_text())
    by_agent: dict[str, list[tuple[int, int]]] = {}
    for s in payload["sessions"]:
        psd = s.get("post_survey_data") or {}
        comfort = COMFORT_MAP.get(psd.get("comfort"))
        rating = RATING_MAP.get(psd.get("rating"))
        if comfort is None or rating is None:
            continue
        by_agent.setdefault(s["interviewer_agent_id"], []).append((comfort, rating))
    return {
        agent: {
            "comfort_level": _mean([c for c, _ in vals]),
            "overall_experience": _mean([r for _, r in vals]),
        }
        for agent, vals in by_agent.items()
    }


def aggregate(eval_data: dict) -> dict:
    """Read each metric straight from its evaluation file's top-level
    aggregate — already a per-participant average in the package's own
    formulas (see src/interviewplayground/interviewreportcard/)."""
    r = eval_data["results"]
    cl = r["conversation_length"]
    rq = r["participant_responses"]
    ib = r["interviewer_behavior"]
    pe = r["participant_experience"]
    return {
        "avg_turns": cl["avg_turns"],
        "avg_response_length": cl["avg_response_length"],
        "relevant_response_volume_avg": rq["relevant_response_volume"],
        "interview_guide_coverage": rq["interview_guide_coverage"],
        "novel_responses_avg": rq["novel_responses"],
        "coherence": ib["coherence"],
        "adaptiveness": ib["adaptiveness"],
        "leading_questions_avg": ib["leading_questions"],
        "support_rapport_avg": ib["support_rapport"],
        "unclear_questions_avg": ib["unclear_questions"],
        "comfort_level": pe["comfort_level"],
        "overall_experience": pe["overall_experience"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="obesity_weight_management",
                        choices=["obesity_weight_management", "asian_american_politics", "genai_knowledge_work"])
    args = parser.parse_args()
    preset = args.preset

    human_agg = {a: aggregate(load_eval(HUMAN_EVAL_DIR, preset, a)) for a in AGENTS}
    sim_agg = {a: aggregate(load_eval(SIM_EVAL_DIR, preset, a)) for a in AGENTS}

    self_report = self_report_by_agent(preset)
    for a in AGENTS:
        for metric in SELF_REPORT_METRICS:
            human_agg[a][metric] = self_report[a][metric]

    by_metric = {}
    for metric in METRICS:
        human_vals = [human_agg[a][metric] for a in AGENTS]
        sim_vals = [sim_agg[a][metric] for a in AGENTS]
        diffs = [s - h for h, s in zip(human_vals, sim_vals)]
        by_metric[metric] = {
            "human_source": "self_report" if metric in SELF_REPORT_METRICS else "llm_judge",
            "pearson_r": _pearson(human_vals, sim_vals),
            "spearman_rho": _spearman(human_vals, sim_vals),
            "mae": _mean([abs(d) for d in diffs]),
            "avg_error": _mean(diffs),  # mean(simulation - human): signed, shows directional bias
            "by_agent": {a: {"human": human_agg[a][metric], "simulation": sim_agg[a][metric]} for a in AGENTS},
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if preset == "obesity_weight_management" else f"__{preset}"
    out_path = OUT_DIR / f"human_vs_simulation_comparison{suffix}.json"
    out_path.write_text(json.dumps({
        "preset": preset, "agents": AGENTS, "metrics": by_metric,
    }, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")

    for metric, s in by_metric.items():
        r = f"{s['pearson_r']:.3f}" if s["pearson_r"] is not None else "n/a"
        rho = f"{s['spearman_rho']:.3f}" if s["spearman_rho"] is not None else "n/a"
        print(f"{metric}: pearson_r={r} spearman_rho={rho} mae={s['mae']:.3f} avg_error(sim-human)={s['avg_error']:+.3f}")


if __name__ == "__main__":
    main()
