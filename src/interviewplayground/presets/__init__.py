"""
Preset studies for testing interviewer agents without running LLM setup workflows.

Usage:
    from interviewplayground import load_preset
    study = load_preset("obesity_weight_management")
    response = study.participants[0].ask("Can you tell me about your experience?")

Available presets:
    obesity_weight_management  — Weight discussions in primary care and commercial referrals (30 participants)
    asian_american_politics    — Asian American identity and political preferences (30 participants)
    genai_knowledge_work       — GenAI tools in knowledge work contexts (30 participants)

Adding a preset:
    1. Save a Study to a JSON file using study.save("my_preset.json").
    2. Move the JSON file into this directory (src/interviewplayground/presets/).
    3. Add an entry to _PRESETS below mapping a name to the filename.
"""

import pathlib

from ..study import Study

_PRESET_DIR = pathlib.Path(__file__).parent

_PRESETS = {
    "obesity_weight_management": _PRESET_DIR / "obesity_weight_management.json",
    "asian_american_politics": _PRESET_DIR / "asian_american_politics.json",
    "genai_knowledge_work": _PRESET_DIR / "genai_knowledge_work.json",
}


def load_preset(name: str) -> Study:
    """
    Return a fully populated Study by preset name.

    No LLM calls are made; all memories are pre-written. The returned Study
    is ready for participant.ask() calls.

    Raises ValueError for unknown preset names.
    """
    if name not in _PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {sorted(_PRESETS)}"
        )
    return Study.load(str(_PRESETS[name]))
