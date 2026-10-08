"""
Pool the three per-study human-vs-simulation comparisons
(obesity_weight_management, asian_american_politics, genai_knowledge_work)
into one aggregate comparison per metric, across all 15 study x agent pairs,
rather than each study's own n=5.
Usage:
    python aggregate_all_studies_comparison.py
"""
from __future__ import annotations

import json
import pathlib

from compare_human_vs_simulation import _pearson, _spearman, _mean, METRICS, AGENTS

BASE_DIR = pathlib.Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "comparisons"

PRESETS = ["obesity_weight_management", "asian_american_politics", "genai_knowledge_work"]


def _comparison_path(preset: str) -> pathlib.Path:
    suffix = "" if preset == "obesity_weight_management" else f"__{preset}"
    return OUT_DIR / f"human_vs_simulation_comparison{suffix}.json"


def main() -> None:
    per_preset = {p: json.loads(_comparison_path(p).read_text()) for p in PRESETS}

    by_metric = {}
    for metric in METRICS:
        points = []
        for preset in PRESETS:
            by_agent = per_preset[preset]["metrics"][metric]["by_agent"]
            for agent in AGENTS:
                v = by_agent[agent]
                points.append({
                    "preset": preset, "agent": agent,
                    "human": v["human"], "simulation": v["simulation"],
                })

        human_vals = [p["human"] for p in points]
        sim_vals = [p["simulation"] for p in points]
        diffs = [s - h for h, s in zip(human_vals, sim_vals)]
        by_metric[metric] = {
            "human_source": per_preset["obesity_weight_management"]["metrics"][metric]["human_source"],
            "n": len(points),
            "pearson_r": _pearson(human_vals, sim_vals),
            "spearman_rho": _spearman(human_vals, sim_vals),
            "mae": _mean([abs(d) for d in diffs]),
            "avg_error": _mean(diffs),
            "points": points,
        }

    out_path = OUT_DIR / "human_vs_simulation_comparison__all_studies.json"
    out_path.write_text(json.dumps({
        "presets": PRESETS, "agents": AGENTS, "metrics": by_metric,
    }, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")
    for metric, s in by_metric.items():
        r = f"{s['pearson_r']:.3f}" if s["pearson_r"] is not None else "n/a"
        rho = f"{s['spearman_rho']:.3f}" if s["spearman_rho"] is not None else "n/a"
        print(f"{metric}: n={s['n']} pearson_r={r} spearman_rho={rho} mae={s['mae']:.3f} avg_error={s['avg_error']:+.3f}")


if __name__ == "__main__":
    main()
