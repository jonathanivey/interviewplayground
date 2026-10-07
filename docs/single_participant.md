# Single-Participant Setup

You can create and interview a single `Participant` directly, without setting up a full `Study`. This is useful for quick testing, prototyping an interviewer agent, or evaluating prompts.

## Minimal Example

```python
from interviewplayground import Participant, Memory

# 1. Create a participant and set their persona and traits
p = Participant()
p.persona = (
    "Alex is a 29-year-old UX researcher at a tech startup. "
    "They are thoughtful, direct, and mildly skeptical of authority. "
    "They recently went through a significant career transition."
)
p.verbosity = "High"
p.reflexivity = "High"
p.disclosure = "Medium"

# 2. Create memory slots
p.create_blank_memories(5)

# 3. Manually mark two memories as target memories
#    (target_indices reference a list you maintain yourself)
p.memories[0].target_indices = [0]
p.memories[1].target_indices = [1]

# 4. Generate target memories (pass your own topic strings)
topics = [
    "What prompted the participant's career transition and their emotional response",
    "What the participant misses about their previous role",
]
p.generate_target_memories(topics)

# 5. Generate non-target memories (fills the remaining 3 blank slots)
p.generate_nontarget_memories()

# 6. Interview the participant
response = p.ask("What made you decide to switch careers?")
print(response)

response = p.ask("Do you ever have second thoughts about the decision?")
print(response)
```

## Inspecting Memories and Transcript

```python
# See all memories
for i, m in enumerate(p.memories):
    print(f"[{i}] target={m.target_indices} reflexive={m.reflexive} sensitive={m.sensitive}")
    print(f"     {m.content[:80]}...")
    print()

# See the full interview transcript
for turn in p.transcript:
    role = turn["role"].capitalize()
    print(f"{role}: {turn['content']}\n")
```

## Manually Writing Memories

You can skip LLM generation and write memories directly — useful for controlled experiments:

```python
from interviewplayground import Participant, Memory

p = Participant()
p.persona = "Jordan is a 45-year-old high school teacher with 20 years of experience."

m1 = Memory()
m1.content = "I started teaching because I had a mentor in high school who changed my life."
m1.target_indices = [0]
m1.reflexive = True
m1.sensitive = False

m2 = Memory()
m2.content = "Last spring I almost quit. I was burned out and couldn't see a way forward."
m2.target_indices = [1]
m2.reflexive = True
m2.sensitive = True

m3 = Memory()
m3.content = "I coach the debate team after school on Tuesdays."
# no target_indices — this is a non-target background memory

p.memories = [m1, m2, m3]

response = p.ask("How did you get into teaching?")
print(response)
```

## Using a Custom LLM Model

Three approaches, applied in priority order:

**1. Per-call `model=` argument** — overrides the default for that call only:

```python
response = p.ask("How did you cope?", model="gpt-5.5")
p.generate_target_memories(topics, model="gpt-5.5")
p.generate_nontarget_memories(model="gpt-5.5")
study = Study.from_description(description, model="gpt-5.5")
```

**2. `set_default_model()`** — sets the default for the entire session. Takes priority over the environment variable:

```python
from interviewplayground import set_default_model
set_default_model("gpt-5.5")
```

**3. `SIMSTUDY_MODEL` environment variable** — applied when neither of the above is set:

```python
import os
os.environ["SIMSTUDY_MODEL"] = "claude-3-5-haiku-20241022"
```

## Notes

- `generate_nontarget_memories()` only fills blank slots where `target_indices == []`. It is safe to call after `generate_target_memories()`.
- `generate_target_memories(topics)` only fills blank slots where `target_indices != []`. The `topics` list should be indexed to match the values in each memory's `target_indices`.
- `ask()` appends every question and response to `p.transcript` in order. The full transcript is included in subsequent prompts, so the participant maintains conversational continuity automatically.
- `ask()` retrieves the most semantically relevant memories for each question via cosine-similarity ranking. The number of memories passed to the prompt is controlled by `p.retrieval_top_k` (default `5`).
