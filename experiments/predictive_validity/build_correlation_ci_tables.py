"""
Build CSV tables of 95% confidence intervals for the human-vs-simulation
Pearson r / Spearman rho reported in pearson_correlation.csv /
spearman_correlation.csv, via the Fisher z-transformation.

Reads the same comparisons/human_vs_simulation_comparison{,__asian_american_politics,
__genai_knowledge_work,__all_studies}.json files as build_correlation_tables.py.

Writes:
    tables/pearson_correlation_ci.csv
    tables/spearman_correlation_ci.csv

Usage:
    python build_correlation_ci_tables.py
"""
from __future__ import annotations

import csv
import json
import math
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
COMPARISONS_DIR = BASE_DIR / "comparisons"
OUT_DIR = BASE_DIR / "tables"

Z_95 = 1.959963985  # two-tailed 95% normal critical value

# (column header, comparison file) — obesity has no filename suffix.
COLUMNS = [
    ("obesity_weight_management", "human_vs_simulation_comparison.json"),
    ("asian_american_politics", "human_vs_simulation_comparison__asian_american_politics.json"),
    ("genai_knowledge_work", "human_vs_simulation_comparison__genai_knowledge_work.json"),
    ("all_studies", "human_vs_simulation_comparison__all_studies.json"),
]

METRICS = [
    "avg_turns", "avg_response_length",
    "relevant_response_volume_avg",
    "interview_guide_coverage",
    "novel_responses_avg",
    "coherence", "adaptiveness",
    "leading_questions_avg", "support_rapport_avg", "unclear_questions_avg",
    "comfort_level", "overall_experience",
]


def _n(entry: dict) -> int:
    return entry["n"] if "n" in entry else len(entry["by_agent"])


def fisher_ci(r: float | None, n: int, se_factor: float) -> tuple[float, float] | None:
    """95% CI for a correlation coefficient via the Fisher z-transformation."""
    if r is None or n < 4:
        return None
    r_clamped = max(-0.999999, min(0.999999, r))
    z = math.atanh(r_clamped)
    se = se_factor / math.sqrt(n - 3)
    lo = math.tanh(z - Z_95 * se)
    hi = math.tanh(z + Z_95 * se)
    return lo, hi


def write_table(stat_key: str, se_factor: float, out_path: pathlib.Path) -> None:
    data = {col: json.loads((COMPARISONS_DIR / fname).read_text()) for col, fname in COLUMNS}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["metric"]
        for col, _ in COLUMNS:
            header += [f"{col}_r", f"{col}_ci_lower", f"{col}_ci_upper", f"{col}_n"]
        writer.writerow(header)
        for metric in METRICS:
            row = [metric]
            for col, _ in COLUMNS:
                entry = data[col]["metrics"][metric]
                r = entry[stat_key]
                n = _n(entry)
                ci = fisher_ci(r, n, se_factor)
                lo, hi = ci if ci is not None else ("", "")
                row += [r if r is not None else "", lo, hi, n]
            writer.writerow(row)

    print(f"wrote {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_table("pearson_r", 1.0, OUT_DIR / "pearson_correlation_ci.csv")
    write_table("spearman_rho", 1.06, OUT_DIR / "spearman_correlation_ci.csv")


if __name__ == "__main__":
    main()
