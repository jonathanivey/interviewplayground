"""
Build CSV tables of the human-vs-simulation correlation per metric, split by
study and pooled across all studies.

Reads comparisons/human_vs_simulation_comparison{,__asian_american_politics,
__genai_knowledge_work,__all_studies}.json (compare_human_vs_simulation.py /
aggregate_all_studies_comparison.py output).

Writes:
    tables/pearson_correlation.csv
    tables/spearman_correlation.csv

Usage:
    python build_correlation_tables.py
"""
from __future__ import annotations

import csv
import json
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
COMPARISONS_DIR = BASE_DIR / "comparisons"
OUT_DIR = BASE_DIR / "tables"

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


def write_table(stat_key: str, out_path: pathlib.Path) -> None:
    data = {col: json.loads((COMPARISONS_DIR / fname).read_text()) for col, fname in COLUMNS}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + [col for col, _ in COLUMNS])
        for metric in METRICS:
            row = [metric] + [data[col]["metrics"][metric][stat_key] for col, _ in COLUMNS]
            writer.writerow(row)

    print(f"wrote {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_table("pearson_r", OUT_DIR / "pearson_correlation.csv")
    write_table("spearman_rho", OUT_DIR / "spearman_correlation.csv")


if __name__ == "__main__":
    main()
