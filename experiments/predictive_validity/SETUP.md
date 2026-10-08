# Running the predictive-validity / robustness experiments

## 1. Install the package

These scripts import `interviewplayground` as a regular installed package —
install it (from the repo root) before running anything here:

```bash
pip install -e ".[dev]"
```

## 2. Repo layout

Everything needed to run an experiment lives in this repo. `experiment_config.py`
(in each experiment folder under `experiments/`) loads the interviewer
agents from `experiments/interviewer_agents/` — reimplementations of the
five AI interviewer products evaluated in the paper (`interviewgpt`,
`llmroleplay`, `mimitalk`, `sparkme`, `storysage`), shared across all
experiment folders. Each study's `interview_description`/`interview_spec` is
likewise sourced locally, from the preset itself.

Comparisons against the real human study (`compare_human_vs_simulation.py`
and the `robustness/*/compare_*_vs_human.py` scripts) are the one exception:
they read from a top-level `human_data/` directory, which is not included in
this release (IRB-protected participant data). Everything else — running
interviews, evaluating them, and the robustness/predictive-validity analysis
that doesn't require the human data — works standalone.

## 3. Secrets

Create a `.env` file at the repo root (never commit this) with:

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

`experiment_config.py`'s `load_env()` reads this file directly.

## 4. Running locally

```bash
cd experiments/predictive_validity
python run_interviews.py --preset obesity_weight_management --agent all --workers 8
python run_evaluation.py --preset obesity_weight_management --agent all
python compare_human_vs_simulation.py --preset obesity_weight_management   # requires human_data/
python build_correlation_tables.py
```

The `experiments/robustness/<model>/` folders follow the same
`run_interviews.py` / `run_evaluation.py` pattern, swapping only the
participant simulator's model (see each folder's `experiment_config.py`).
The `gemma`/`qwen`/`qwen4b`/`qwen08b` variants require a self-hosted vLLM
server already running and reachable at `PARTICIPANT_API_BASE`.

Both `run_interviews.py` and `run_evaluation.py` are I/O-bound (waiting on API
round-trips), not compute-bound, and need no GPU — except the self-hosted
robustness variants, whose vLLM server must be launched separately on a GPU
node first.
