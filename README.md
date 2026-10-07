# interviewplayground

A Python package for creating AI-simulated interview participants to test and evaluate qualitative research interviewer agents. Each simulated participant holds a persona, behavioral traits, and a memory system. Call `participant.ask(question)` to get a natural-language response grounded in that participant's persona and memories. The package also ships InterviewReportCard, an evaluation suite that scores interview transcripts so you can measure how well your own AI interviewer is performing.

## Installation

```bash
pip install interviewplayground
```

Or install from source in development mode:

```bash
git clone <repo-url>
cd interviewplayground
pip install -e ".[dev]"
```

## Setup

The package uses [LiteLLM](https://github.com/BerriAI/litellm), so it works with any provider LiteLLM supports — including OpenAI, Anthropic, Google Gemini, and self-hosted models via vLLM.

### API-based models (OpenAI, Anthropic, Gemini)

Set your API key and, if not using the default OpenAI model, the model name:

```bash
# OpenAI (default — no model override needed)
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
export SIMSTUDY_MODEL="claude-3-5-haiku-20241022"

# Google Gemini
export GEMINI_API_KEY="..."
export SIMSTUDY_MODEL="gemini/gemini-2.0-flash"
```

Or set it programmatically:

```python
from interviewplayground import set_default_model

set_default_model("claude-3-5-haiku-20241022")
```

> **Note on embeddings with non-OpenAI models:** Anthropic and Gemini models do not support the embeddings API. Keep `SIMSTUDY_EMBEDDING_MODEL` pointed at an OpenAI embedding model (the default `text-embedding-3-small`) and set `OPENAI_API_KEY` even when your completion model is Claude or Gemini.

### Self-hosted models via vLLM

vLLM is a first-class target for this package — both the participant simulator and the InterviewReportCard judge can run against a self-hosted model, not just hosted APIs.

#### Running a local vLLM server

```bash
vllm serve <model-name> --host 0.0.0.0 --port 8000
```

```python
from interviewplayground import set_default_model

set_default_model(
    "openai/<model-name>",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",  # vLLM requires a non-empty string; the value is ignored
)
```

`extra_body` is forwarded as-is to every completion call, which is where model-specific serving flags go (e.g. a model's `chat_template_kwargs`):

```python
set_default_model(
    "openai/Qwen/Qwen3.5-9B",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

Or configure the participant model via environment variables:

```bash
export SIMSTUDY_MODEL="hosted_vllm/my-model-name"
export SIMSTUDY_API_BASE="http://localhost:8000/v1"
export SIMSTUDY_API_KEY="EMPTY"
```

#### Using vLLM as the InterviewReportCard judge

The same `api_base`/`extra_body` pattern applies when evaluating interviews — pass them through to `Study.evaluate()` (or any individual `evaluate_*` function):

```python
study.evaluate(
    model="openai/<judge-model-name>",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
)
```

`use_batch=True` on any `evaluate_*` function works against a vLLM server too: since a local server has no provider batch API, it automatically falls back to running prompts concurrently over `call()` instead (as opposed to the native OpenAI/Azure/Gemini batch APIs used for those providers). Concurrency defaults to the `LLM_BATCH_CONCURRENCY` environment variable (16 if unset) — set it to roughly your server's `--max-num-seqs`.

#### Using different models for setup vs. interviews

Memory generation and study setup require instruction-following and structured JSON output — use an API model for those. The `ask()` call only needs conversational generation, making it a good fit for a locally-served model. You can mix both in the same session with per-call overrides:

```python
from interviewplayground import set_default_model

# Default: API model for setup (memory generation, study description)
set_default_model("gpt-4o-mini")

# Per-call override on ask() to use a local vLLM model
response = p.ask(
    "How did you cope?",
    model="hosted_vllm/osim-8b",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
)
```

### Embedding model

The embedding model is configured separately from the completion model. It defaults to `text-embedding-3-small` (OpenAI). To change it:

```python
from interviewplayground import set_embedding_model

set_embedding_model("text-embedding-3-small")  # default

# Or point to a vLLM-hosted embedding model
set_embedding_model(
    "hosted_vllm/my-embedding-model",
    api_base="http://localhost:8001/v1",
    api_key="EMPTY",
)
```

Or via environment variables:

```bash
export SIMSTUDY_EMBEDDING_MODEL="text-embedding-3-small"
export SIMSTUDY_EMBEDDING_API_BASE="http://localhost:8001/v1"  # only needed for vLLM embeddings
export SIMSTUDY_EMBEDDING_API_KEY="EMPTY"
```

### Model selection priority

For the completion model, priority is highest to lowest:

1. **Per-call `model=` argument** — overrides everything for that one call
2. **`set_default_model()`** — programmatic session-wide default
3. **`SIMSTUDY_MODEL` env var** — environment-level fallback
4. Built-in default (`gpt-5-mini`)

`api_base` and `api_key` follow the same priority order, using `SIMSTUDY_API_BASE` and `SIMSTUDY_API_KEY` as their env var fallbacks. The embedding model follows the same pattern via `set_embedding_model()` and `SIMSTUDY_EMBEDDING_*` env vars.

All LLM-calling methods accept per-call overrides:

```python
p.ask("How did you cope?", model="...", api_base="...", api_key="...")
p.generate_target_memories(topics, model="...", api_base="...", api_key="...")
p.generate_nontarget_memories(model="...", api_base="...", api_key="...")
Study.from_description(description, model="...", api_base="...", api_key="...")
```

> **Note on memory generation:** Generating 185 non-target memories in a single batch can take several minutes. The default LLM timeout is 600 seconds.

---

## Full Workflow

### Step 1 — Generate target information from a study description

```python
from interviewplayground import Study

study = Study.from_description("""
    A grounded theory study examining how informal caregivers of adults with dementia
    manage their own wellbeing while providing care. We are interested in the strategies
    they use, the support networks they rely on, and how their identity changes over time.
""")

print(study.target_information)
# ['How caregivers first took on the caregiving role and whether it was a choice',
#  'Daily routines caregivers use to maintain their own mental health', ...]
```

Or supply your own list directly:

```python
study = Study(target_information=[
    "How caregivers first took on the caregiving role",
    "Support networks caregivers rely on",
    ...
])
```

### Step 2 — Create participants

```python
study.create_participants(3)
```

This creates 3 `Participant` objects with default personas and `"Medium"` traits.

### Step 3 — Set personas and traits, create blank memories

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

### Step 4 — Distribute target information

```python
study.distribute_target_information(
    avg_per_participant=4.0,   # average number of target items each participant knows
    ensure_all_distributed=True  # guarantee every target appears in at least one participant
)
```

This assigns target indices to blank memory slots across participants. Participants will know about different subsets of the target information — like real study participants.

### Step 5 — Generate non-target memories

```python
for p in study.participants:
    p.generate_nontarget_memories()
```

One LLM call per participant. Fills blank non-target memory slots with autobiographical content consistent with the persona and traits.

### Step 6 — Generate target memories

```python
for p in study.participants:
    p.generate_target_memories(study.target_information)
```

One LLM call per participant. Fills blank target memory slots with memories grounded in the corresponding target information items.

### Step 7 — Interview participants

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

---

## Evaluating Interview Quality

Once a study's participants have transcripts (built up via `ask()`, as above), `study.evaluate()` runs the InterviewReportCard suite over them — no separate setup needed.

```python
results = study.evaluate()
print(results.keys())
# dict_keys(['conversation_length', 'participant_responses', 'interviewer_behavior', 'participant_experience'])
```

| Dimension | Metrics |
|---|---|
| `conversation_length` *(no LLM calls)* | `avg_turns`, `avg_response_length` |
| `participant_responses` | `relevant_response_volume`, `interview_guide_coverage`, `novel_responses` |
| `interviewer_behavior` | `coherence`, `adaptiveness`, `leading_questions`, `support_rapport`, `unclear_questions` |
| `participant_experience` | `comfort_level`, `overall_experience` |

Every dimension's dict also includes a `per_participant` list with each participant's own unaggregated value(s), alongside the group-level average. `research_questions`/`interview_guide` default to the Study's own attributes if not passed explicitly.

For large evaluation runs, `use_batch=True` (on `study.evaluate()` or any individual `evaluate_*` function) submits prompts as provider batch jobs instead of one call at a time. For submit-now/retrieve-later workflows (e.g. across process restarts), each dimension also exposes `submit_*`/`retrieve_*` functions directly — see `interviewplayground.interviewreportcard`. Note the native batch APIs are OpenAI/Azure/Gemini-only; see "Using vLLM as the InterviewReportCard judge" above for the local-model equivalent.

### Running an evaluation against your own interviewer

To evaluate your own AI interviewer, drive the interview loop yourself, calling your interviewer to produce each question and `participant.ask()` to produce each response:

```python
participant = study.participants[0]
transcript_so_far = []

while not my_interviewer.is_done(transcript_so_far):
    question = my_interviewer.next_question(transcript_so_far)
    answer = participant.ask(question)
    transcript_so_far.append({"role": "interviewer", "content": question})
    transcript_so_far.append({"role": "participant", "content": answer})

# Repeat for every participant, then:
results = study.evaluate()
```

`participant.ask()` already appends each turn to `participant.transcript`, so `study.evaluate()` picks it up automatically once every participant has been interviewed.

---

## Using Presets

Three pre-built study presets are included for immediate testing — no LLM calls needed to load them.

```python
from interviewplayground import load_preset

study = load_preset("obesity_weight_management")

# Participants are ready to interview
p = study.participants[0]
print(p.persona)

response = p.ask("How have weight conversations with your doctor gone?")
print(response)
```

Available presets:

| Name | Topic | Participants |
|---|---|---|
| `obesity_weight_management` | Weight discussions in primary care and commercial program referrals | 30 |
| `asian_american_politics` | Asian American identity and political preferences | 30 |
| `genai_knowledge_work` | GenAI tools in knowledge work contexts | 30 |

Each participant has 200 memories (15 target memories covering the study's research questions, plus autobiographical background memories).

### Adding a preset

1. Build and fully populate a `Study` object (with participants and memories).
2. Call `study.save("my_preset.json")` and move the file into `src/interviewplayground/presets/`.
3. In `src/interviewplayground/presets/__init__.py`, add one entry to `_PRESETS`:
   ```python
   _PRESETS = {
       ...
       "my_preset": _PRESET_DIR / "my_preset.json",
   }
   ```

---

## Customizing Prompts

All LLM prompts are string constants in [`src/interviewplayground/prompts.py`](src/interviewplayground/prompts.py). Edit them directly to change how memories are generated or how participants respond in interviews.

The prompts use standard Python `.format()` placeholders:

| Prompt | Placeholders |
|---|---|
| `STUDY_FROM_DESCRIPTION` | `{description}` |
| `NONTARGET_MEMORIES` | `{persona}`, `{knowledge}`, `{verbosity}`, `{memory}`, `{reflexivity}`, `{disclosure}`, `{understanding}`, `{n}` |
| `TARGET_MEMORIES` | same as above plus `{topics_list}`, `{n}` |
| `ASK` | same traits plus `{retrieved_memories}`, `{transcript}`, `{question}` |

---

## Running Tests

Tests do not require an API key — LLM calls are mocked.

```bash
pytest
```

---

## Single-Participant Setup

See [docs/single_participant.md](docs/single_participant.md) for how to set up and interview a single participant without a full Study.
