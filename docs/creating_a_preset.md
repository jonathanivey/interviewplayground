# Creating a Custom Study or Preset

Most users should start with the built-in presets (see the main [README](../README.md)). This guide is for building a `Study` from scratch — for a new interview topic, a new participant population, or a custom preset you want to reuse.

## Step 1 — Identify insights

Manually identify the key findings that you want the simulated participants to be able to produce.

```python
from interviewplayground import Study

study = Study(insights=[
    "How caregivers first took on the caregiving role",
    "Support networks caregivers rely on",
    ...
])
```

## Step 2 — Create participants

```python
study.create_participants(3)
```

This creates 3 `Participant` objects with default personas and `"Medium"` traits.

## Step 3 — Set personas and traits, create blank memories

```python
p = study.participants[0]

p.persona = (
    "Maria is a 52-year-old woman who left her part-time job three years ago to care "
    "for her mother, who has moderate Alzheimer's disease. She is warm and reflective "
    "but carries significant grief about her mother's decline."
)
p.knowledge = "Medium"
p.verbosity = "High"
p.memory = "High"
p.reflexivity = "High"
p.disclosure = "Medium"
p.understanding = "Medium"

p.create_blank_memories(12)
```

Valid trait values: `"Low"`, `"Medium"`, `"High"`.

Repeat for each participant.

## Step 4 — Distribute insights

```python
study.distribute_insights(
    avg_per_participant=4.0,   # average number of insights each participant knows
    ensure_all_distributed=True  # guarantee every insight appears in at least one participant
)
```

This assigns insight indices to blank memory slots across participants. Participants will know about different subsets of the study's insights.

## Step 5 — Generate background memories

```python
for p in study.participants:
    p.generate_background_memories()
```

One LLM call per participant. Fills blank non-insight memory slots with autobiographical content consistent with the persona and traits.

## Step 6 — Generate insight memories

```python
for p in study.participants:
    p.generate_insight_memories(study.insights)
```

One LLM call per participant. Fills blank insight memory slots with memories grounded in the assigned insights.

## Step 7 — Interview participants

```python
response = study.participants[0].ask("Can you tell me a bit about how you ended up in a caregiving role?")
print(response)
```

Each call adds the question and response to `participant.transcript`. The participant's memories and full transcript context are included in the prompt automatically.

```python
# Inspect the transcript
for turn in study.participants[0].transcript:
    print(f"{turn['role'].capitalize()}: {turn['content']}\n")
```

## Adding research questions and an interview guide

Presets include `research_questions` and `interview_guide` so InterviewReportCard and any AI interviewer you test can use them. Set them directly on your `Study`:

```python
study.research_questions = [
    "How do caregivers first take on the caregiving role, and is it a choice?",
    "What support networks do caregivers rely on?",
]

study.interview_guide = [
    {
        "topic": "Becoming a Caregiver",
        "subtopics": [
            "The circumstances that led to taking on the caregiving role",
            "Whether the role felt like a choice or an obligation",
        ],
    },
    {
        "topic": "Support Networks",
        "subtopics": [
            "Family members or friends who provide support",
            "Use of formal support services or caregiver groups",
        ],
    },
]
```

## Registering it as a reusable preset

Once a `Study` is fully populated (participants, memories, research questions, and an interview guide), save it and register it so `load_preset()` can find it:

1. Call `study.save("my_preset.json")` and move the file into `src/interviewplayground/presets/`.
2. In `src/interviewplayground/presets/__init__.py`, add one entry to `_PRESETS`:
   ```python
   _PRESETS = {
       ...
       "my_preset": _PRESET_DIR / "my_preset.json",
   }
   ```

`load_preset("my_preset")` will now load it with no further LLM calls.
