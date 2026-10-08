"""
Build a CSV table of pooled (all_studies) Pearson correlation with the human
studies, across the seven simulator models tested in this project:
gemini-3.1-pro-preview (the predictive_validity baseline), gpt-5.6-terra,
Gemma-4-31B-AWQ, gemini-3.7-flash, Qwen/Qwen3.5-9B, Qwen/Qwen3.5-4B, and
Qwen/Qwen3.5-0.8B.

Writes:
    tables/pearson_vs_human.csv

Usage:
    python build_correlation_tables.py
"""
from __future__ import annotations

import csv
import json
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
OUT_DIR = BASE_DIR / "tables"

VS_HUMAN_FILES = {
    "gemini": REPO_ROOT / "experiments" / "predictive_validity" / "comparisons" / "human_vs_simulation_comparison__all_studies.json",
    "gpt": BASE_DIR / "gpt" / "comparisons" / "gpt_vs_human_comparison__all_studies.json",
    "gemma": BASE_DIR / "gemma" / "comparisons" / "gemma_vs_human_comparison__all_studies.json",
    "flash": BASE_DIR / "flash" / "comparisons" / "flash_vs_human_comparison__all_studies.json",
    "qwen": BASE_DIR / "qwen" / "comparisons" / "qwen_vs_human_comparison__all_studies.json",
    "qwen4b": BASE_DIR / "qwen4b" / "comparisons" / "qwen4b_vs_human_comparison__all_studies.json",
    "qwen08b": BASE_DIR / "qwen08b" / "comparisons" / "qwen08b_vs_human_comparison__all_studies.json",
}
METRICS = [
    "avg_turns", "avg_response_length",
    "relevant_response_volume_avg",
    "interview_guide_coverage",
    "novel_responses_avg",
    "coherence", "adaptiveness",
    "leading_questions_avg", "support_rapport_avg", "unclear_questions_avg",
    "comfort_level", "overall_experience",
]


def write_table(files: dict[str, pathlib.Path], out_path: pathlib.Path) -> None:
    data = {model: json.loads(path.read_text()) for model, path in files.items()}
    models = list(files)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + models)
        for metric in METRICS:
            row = [metric] + [data[model]["metrics"][metric]["pearson_r"] for model in models]
            writer.writerow(row)

    print(f"wrote {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_table(VS_HUMAN_FILES, OUT_DIR / "pearson_vs_human.csv")


if __name__ == "__main__":
    main()
