"""
Shared configuration for the robustness experiment: does InterviewPlayground's predictive validity hold up when the participant
simulator is swapped from Gemini to a self-hosted Gemma-4 (AWQ 4-bit)?

Same presets, interviewer agents, and judge as
experiments/robustness/gpt/experiment_config.py; only
PARTICIPANT_MODEL/PARTICIPANT_API_BASE/PARTICIPANT_API_KEY differ.

The participant model is served by vLLM on a self-hosted GPU node (see
launch_vllm_gemma.slurm in your vLLM deployment) — an OpenAI-compatible
endpoint, hence the "openai/" litellm provider prefix and
PARTICIPANT_API_BASE pointing at that node's :8000/v1. The judge stays on
Gemini (its own Batch API, unaffected by the participant's api_base).
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent

INTERVIEWER_AGENTS_DIR = REPO_ROOT / "experiments"

for _path in (INTERVIEWER_AGENTS_DIR,):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# preset name -> a stable per-study id (passed through to the interviewer
# agents as study_id; the agents don't use it for lookup).
PRESET_TO_STUDY = {
    "obesity_weight_management": "study_001",
    "asian_american_politics": "study_002",
    "genai_knowledge_work": "study_003",
}
PRESETS = list(PRESET_TO_STUDY)

# One-sentence description of what each study is about, shown to the
# interviewer agents as interview_description.
INTERVIEW_DESCRIPTIONS = {
    "obesity_weight_management": (
        "their perceptions and expectations about obesity treatment in "
        "primary care and referral to community-commercial sector programs"
    ),
    "asian_american_politics": (
        "their ethnic origin, identity, and how it shapes their political "
        "preferences"
    ),
    "genai_knowledge_work": (
        "their information management practices, collaboration challenges, "
        "and attitudes towards future AI assistance"
    ),
}

# The five real interviewer products used in the human study.
AGENTS = ["interviewgpt", "llmroleplay", "mimitalk", "sparkme", "storysage"]

# The participant simulator runs on a self-hosted Gemma-4-31B-it-AWQ-4bit via
# vLLM (OpenAI-compatible API). Update PARTICIPANT_API_BASE if the server is
# relaunched on a different node — check with:
#   ssh <cluster-login-node> 'squeue -u $(whoami) -n gemma-vllm-server -o "%N %T"'
PARTICIPANT_MODEL = "openai/cyankiwi/gemma-4-31B-it-AWQ-4bit"
PARTICIPANT_API_BASE = "http://<vllm-host>:8000/v1"
PARTICIPANT_API_KEY = "EMPTY"

# Judge stays on Gemini, same as every other predictive-validity/robustness run.
JUDGE_MODEL = "gemini/gemini-3.1-pro-preview"
# Interviewer model is NOT overridden here — all five agent modules hard-code
# DEFAULT_MODEL = "gpt-5.4-mini" already, matching the human study.


def load_env(path: str | os.PathLike | None = None) -> None:
    """Load KEY=VALUE lines from the repo .env into os.environ (setdefault)."""
    env_path = pathlib.Path(path) if path else REPO_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def _resolve_preset(preset: str) -> str:
    if preset not in PRESET_TO_STUDY:
        raise ValueError(f"Unknown preset '{preset}'. Available presets: {PRESETS}")
    return preset


def get_study_config(preset: str) -> dict:
    """Return interview_description and interview_spec for a preset.

    interview_spec is the preset's own interview_guide (topic/subtopic list),
    used both to configure the interviewer agent and as the evaluation
    interview_guide.
    """
    _resolve_preset(preset)
    from interviewplayground import load_preset

    return {
        "interview_description": INTERVIEW_DESCRIPTIONS[preset],
        "interview_spec": load_preset(preset).interview_guide,
    }


def get_interviewer_class(agent_id: str):
    """Return the InterviewerAgent class for one of the five supported agents."""
    if agent_id not in AGENTS:
        raise ValueError(f"Unknown agent '{agent_id}'. Available agents: {AGENTS}")
    module = importlib.import_module(f"interviewer_agents.{agent_id}")
    return module.InterviewerAgent
