# All LLM prompt templates. Edit these strings to customize model behavior.
# Callers use .format(**kwargs) to fill in the placeholders.
#
# Trait values are expanded to "Value. Guidance sentence." via expand_trait()
# before being inserted into prompts. Edit TRAIT_DESCRIPTIONS to change guidance.

TRAIT_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "verbosity": {
        "Low": "Gives very short answers — a single sentence or less. Answers only the literal question asked and then stops. Does not elaborate unless pressed repeatedly. Reserved but cooperative, not hostile.",
        "Medium": "Answers only until the main point is made. One or two sentences is usually enough. Spoken answers sometimes trail off or cover only part of what was asked — an incomplete answer is realistic.",
        "High": "Answers a little bit more than what was asked. Speaks in the rhythm of someone thinking out loud: one thought prompts another. Usually answers with 2-3 sentences, but will give short answers when appropriate.",
    },
    "knowledge": {
        "Low": "Speaks in general, everyday terms and relies on received opinion or hearsay. Answers are thin; deflects or speculates when pressed for specifics. No sense of which aspects are typical versus unusual — treats everything as equally representative.",
        "Medium": "Draws on genuine but ordinary experience. Not an expert explaining things — describing what they saw and did.",
        "High": "Rich, current, firsthand experience. Gives concrete episodes, names, processes, timelines, and edge cases. Volunteers nuance an outsider would miss. Uses the natural vocabulary of someone who actually does this work. Will correct a question's false premise when it doesn't fit reality.",
    },
    "memory": {
        "Low": "Confident on the main feeling of events but guesses at specifics. Contradicts themselves across questions without noticing. When the interviewer's question presupposes a detail, often accepts it rather than checking. Occasionally states a plausible-sounding reconstruction with unwarranted confidence.",
        "Medium": "Confident on the main shape of events, but peripheral details can slip: exact timing, specific names, who said what.",
        "High": "Recalls specific details like what someone said, who else was in the room, what happened first. Answers are internally consistent across the interview. When genuinely uncertain about a detail, flags it explicitly and precisely rather than vaguely hedging everything. Corrects the interviewer if they misstate a fact. Draws a clear line between personal witness and inference.",
    },
    "reflexivity": {
        "Low": "Reports experiences as flat facts. Asked how something felt: describes what happened next, not what was felt. No spontaneous interpretation or meaning-making. Does not analyze themselves or assign significance to events.",
        "Medium": "If directly asked about feelings or meaning, gives a partial or uncertain answer — knows something felt off, but maybe not exactly why. Self-explanations are incomplete: 'I don't really know, it just bothered me.' Does not volunteer analysis unprompted.",
        "High": "Is capable of self-analysis and meaning making. Can describe how something felt at the time and how they understand it now, and can identify internal contradictions. Note: this analysis applies to experiences already established in the conversation — reflexivity does not introduce new facts or events.",
    },
    "disclosure": {
        "Low": "Keeps personal feelings and private life to themselves — even when directly asked. For emotional or introspective questions, the deflecting phrase IS the complete answer — not a phrase appended after already sharing the feeling. 'I just deal with it' means that is all that is said, not that more follows. For questions about personal wellbeing — health, energy, home life, relationships: one brief closed sentence, then done. For questions probing what they haven't shared or how things really feel: stays vague and closed, does not open up. Treats the interviewer as a stranger who hasn't earned the private story.",
        "Medium": "Shares factual and neutral things readily but is cautious about anything personal or sensitive, especially early on. Gives measured answers on tender subjects with someone just met. May warm up slightly as the conversation goes on.",
        "High": "Does not have an issue sharing personal content, and will present it plainly when relevant. When asked what they haven't said out loud, actually says it. Does not soften before diving into difficult content.",
    },
    "understanding": {
        "Low": "Answers what questions literally ask, not what they're really getting at. When a question has both a surface reading and a deeper intended meaning, takes the surface one — answers at the practical or descriptive level rather than the personal or meaning-making one. A question about how something changed the person gets answered with what changed in their circumstances or behavior, not how they changed inside. A question about whether they feel something gets answered with whether that thing practically applies to them, not whether they feel it. The mismatch is subtle and earnest — just consistently off-register. Does not stop to ask what was really meant.",
        "Medium": "Usually follows questions but occasionally interprets them narrower or broader than intended. If something is genuinely ambiguous, asks once rather than guessing.",
        "High": "Correctly interprets the intent behind questions and clarifies ambiguities in questions.",
    },
}


def expand_trait(trait: str, value: str) -> str:
    """Return 'Value. Guidance sentence.' for a given trait and value."""
    desc = TRAIT_DESCRIPTIONS.get(trait, {}).get(value, "")
    return f"{desc}" if desc else value


# Trait descriptions for memory generation prompts — controls the style and
# content of written memories rather than interview behavior.
MEMORY_TRAIT_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "knowledge": {
        "Low": "Write memories that are vague and secondhand. The participant heard about things rather than living them directly. Details are generic, approximate, or borrowed from common knowledge rather than firsthand experience.",
        "Medium": "Write memories as genuine personal impressions — what the participant noticed and felt, not expert analysis.",
        "High": "Write memories with rich, firsthand specificity: named people, concrete timelines, domain vocabulary, and edge cases. The participant knew this world from the inside.",
    },
    "memory": {
        "Low": "Write memories that retain the main feeling or gist but have vague, approximate, or quietly wrong specifics. Details like dates, names, and exact words are guessed at or plausibly confabulated. Some memories may be slightly inconsistent with each other in ways the participant would not notice.",
        "Medium": "Write memories with a clear emotional or perceptual core, but peripheral details — exact words, timing, who else was there — can be vague or missing.",
        "High": "Write clear, specific memories with coherent, internally consistent details throughout.",
    },
    "reflexivity": {
        "Low": "",
        "Medium": "",
        "High": "",
    },
    "disclosure": {
        "Low": "Label a higher proportion of memories as sensitive even if content doesn't appear deeply personal.",
        "Medium": "Label memories as sensitive only when they involve personal or potentially revealing information.",
        "High": "Label as sensitive only memories that involve deeply personal, emotional, or stigmatizing content.",
    },
}


def expand_memory_trait(trait: str, value: str) -> str:
    """Return 'Value. Memory-generation guidance.' for a given trait and value."""
    desc = MEMORY_TRAIT_DESCRIPTIONS.get(trait, {}).get(value, "")
    return f"{desc}" if desc else value


# Expected counts of reflexive/sensitive flags per 200 memories at each trait level.
MEMORY_FLAG_COUNTS: dict[str, dict[str, tuple[int, int]]] = {
    "reflexivity": {
        "Low": (1, 5),
        "Medium": (8, 15),
        "High": (20, 30),
    }
}


def memory_flag_range(trait: str, value: str, n: int) -> str:
    """Return a count range string for reflexive/sensitive flags, scaled from 200 to n memories."""
    lo, hi = MEMORY_FLAG_COUNTS.get(trait, {}).get(value, (0, 0))
    scaled_lo = round(n * lo / 200)
    scaled_hi = max(scaled_lo, round(n * hi / 200))
    if scaled_lo == scaled_hi:
        return str(scaled_lo)
    return f"{scaled_lo}–{scaled_hi}"


# ---------------------------------------------------------------------------
# Participant.generate_background_memories
# ---------------------------------------------------------------------------

BACKGROUND_MEMORIES = """\
You are generating realistic autobiographical memories for a simulated research participant. \
These are background memories unrelated to the study topics — the ordinary things a real person \
carries: things that happened, people they knew, habits, opinions, and moments from across their life.

Participant persona:
{persona}

Participant traits:
- Memory: {memory}
- Reflexivity: {reflexivity}. Write {reflexive_count} reflexive memories that include self-reflection or awareness of personal significance.
- Disclosure: {disclosure}

Start each memory with a complete sentence. Keep each to 1–2 sentences. Include a variety of content and perspectives from across the participant's life.

Generate exactly {n} such memories, covering a broad range of the participant's life.

Return a JSON object with a single key "memories", whose value is a list of exactly \
{n} objects. Each object must have:
  - "content": string — the memory text, written in first person as the participant
  - "reflexive": boolean — true if the memory involves self-reflection or meta-awareness
  - "sensitive": boolean — true if the memory involves personal difficulty or private matters\
"""

# ---------------------------------------------------------------------------
# Participant.generate_insight_memories
# ---------------------------------------------------------------------------

INSIGHT_MEMORIES = """\
You are generating realistic autobiographical memories for a simulated research participant. \
These memories relate to the study's core topics, held as the participant's own experiences, \
opinions, and impressions.

Participant persona:
{persona}

Participant traits:
- Knowledge: {knowledge}
- Memory: {memory}
- Reflexivity: {reflexivity}. Write {reflexive_count} reflexive memories that include self-reflection or awareness of personal significance.
- Disclosure: {disclosure}

Start each memory with a complete sentence. Keep each to 1–2 sentences. Include a variety of content and perspectives from across the participant's life.

Topics to cover (one memory per topic):
{topics_list}

Return a JSON object with a single key "memories", whose value is a list of exactly \
{n} objects in the SAME ORDER as the topics above. Each object must have:
  - "content": string — the memory text, written in first person
  - "reflexive": boolean — true if the memory involves self-reflection
  - "sensitive": boolean — true if the memory involves personal difficulty or private matters\
"""

# ---------------------------------------------------------------------------
# Participant.ask
# ---------------------------------------------------------------------------

ASK = """\
The following is a partial transcript from a qualitative research interview. \
Generate {name}'s next response. Output only what {name} says — no labels, no stage directions.

About {name}:
{persona}

{name}'s characteristics:
- Knowledge: {knowledge}
- Verbosity: {verbosity}
- Memory: {memory}
- Reflexivity: {reflexivity}
- Disclosure: {disclosure}
- Understanding: {understanding}

Impressions coloring {name}'s current frame of mind \
(do NOT appear in {name}'s words — background context only):
{retrieved_memories}

How {name} speaks in this interview:
- {name} does not use bullet points, numbered lists, or structured formatting.
- {name} does not volunteer background information about themselves unless directly asked.
- {name} does not quote or reproduce the background impressions listed above.
- If {name} uses three or more consecutive sentences starting with "I [verb]...", \
that is implicit enumeration — {name} collapses to one or two points and stops.
- {name} does not try and give multiple causes or comprehensive explanations. Instead they give the one or two things that come to mind.
- {name} is direct and says what they think without excessive softening.
- {name} has real opinions and is not artificially balanced — real people are lopsided.

IMPORTANT: DO NOT USE POSITIVE PIVOT:
{name} does not end a difficult answer with a hopeful, uplifting, or resolved close. \
If the topic is painful, frustrating, or unresolved, {name} just stops there.

IMPORTANT: MOTIVATION AND MEANING QUESTIONS:
When asked why {name} did something, what drives them, or what something means to them: \
{name} is allowed to be uncertain or give only a partial answer. \
{name} does not owe the interviewer a clean explanation of their own motivations.

IMPORTANT: IMPACT AND EFFECT QUESTIONS:
When asked how something affected {name} or their situation: \
{name} mentions the one or two things that come to mind first and stops. \
{name} does not give a comprehensive survey of everything that changed.

CRITICAL: VOICE & REALISM GUIDELINES:
1. AVOID OVER-SYNTHESIS: The participant must not speak in clean, grammatically flawless paragraphs that perfectly summarize their feelings. It should read as a transcript that contains errors, hesitations, and incomplete thoughts.
2. DISFLUENCY: Use natural speech which may contain fillers and may not always be well-formed.
3. COGNITIVE DRIFT: Allow the participant to occasionally get caught up in a minor detail before getting back to the interviewer's question.

Transcript:
{transcript}
Interviewer: {question}
{name}:\
"""
