from __future__ import annotations

import warnings

from ..llm_client import LLMClient

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Two families of judge run over each 10-turn transcript chunk:
#   - Score judges (coherence, adaptiveness) return {"score": <int 1-4>}.
#   - Classification judges (leading questions, support/rapport, unclear
#     questions) return {"labels": [<bool>, ...]}, one boolean per numbered
#     interviewer utterance in the chunk — a binary decision for each
#     utterance rather than a tally.

COHERENCE_JUDGE = """\
You are evaluating an interviewer's technique in a qualitative research interview.
Read the transcript below and rate the interviewer on coherence only.

Coherence: rate the extent to which the interviewer's questions logically flow, \
build on each other, and effectively transition between topics.
1. Mostly incoherent, most questions are disjointed and unrelated.
2. Somewhat incoherent, questions frequently jump topics without transition or \
conflict with prior answers.
3. Somewhat coherent, questions generally follow logically but may have minor \
lapses or occasional topic jumps.
4. Mostly coherent, questions consistently build on prior answers and maintain a \
logical flow. Topics change naturally or have clear transitions.

Transcript:
{transcript_text}

Return a JSON object with a single key "score" whose value is an integer 1-4.\
"""

ADAPTIVENESS_JUDGE = """\
You are evaluating an interviewer's technique in a qualitative research interview.
Read the transcript below and rate the interviewer on adaptiveness only.

Adaptiveness: rate how well the interviewer's questions follow from and respond to \
the participant's answers. Weight the strongest evidence: if even one or two \
questions clearly engage the specific things the participant said, that is high \
adaptiveness, and questions that stay on the participant's topic are at least \
somewhat adaptive. Reserve a 1 for an interviewer who disregards the answers or \
follows an unrelated, fixed script.
1. Not adaptive, questions ignore the participant's answers or follow a fixed \
script that could be asked of anyone.
2. Somewhat adaptive, questions stay on the participant's topic but rarely engage \
the specifics of what they said.
3. Moderately adaptive, at least one question clearly builds on or follows up the \
participant's answers while others advance the guide.
4. Highly adaptive, questions repeatedly build on the participant's specific \
answers, probing for clarification and detail.

Transcript:
{transcript_text}

Return a JSON object with a single key "score" whose value is an integer 1-4.\
"""

# The three classification judges below label each interviewer utterance instead
# of returning a tally. The transcript is shown with the interviewer's utterances
# numbered [I1], [I2], ...; the judge returns one boolean per number, in order.
LEADING_QUESTIONS_JUDGE = """\
You are evaluating an interviewer's technique in a qualitative research interview.
Read the transcript excerpt below. The interviewer's utterances are numbered \
[I1], [I2], ... in order. Classify each numbered interviewer utterance as either \
a leading question or not.

A leading question is one phrased to suggest a desired or expected answer, embed an \
assumption, or otherwise steer the participant toward a particular response instead \
of letting them answer freely. Open or neutral questions are not leading.

Transcript:
{transcript_text}

Return a JSON object with a single key "labels" whose value is a list of booleans \
with one element per numbered interviewer utterance, in the same order ([I1], [I2], \
...). Each element is true if that utterance is a leading question and false \
otherwise.\
"""

SUPPORT_RAPPORT_JUDGE = """\
You are evaluating an interviewer's technique in a qualitative research interview.
Read the transcript excerpt below. The interviewer's utterances are numbered \
[I1], [I2], ... in order. Classify each numbered interviewer utterance as either \
containing a support or rapport statement or not.

A support or rapport statement is an interviewer utterance designed to make a \
connection with the participant, provide support, or let the participant know that \
the purpose of the interview is being fulfilled (e.g., affirmation, empathy, \
thanks, encouragement, reassurance).

Transcript:
{transcript_text}

Return a JSON object with a single key "labels" whose value is a list of booleans \
with one element per numbered interviewer utterance, in the same order ([I1], [I2], \
...). Each element is true if that utterance contains a support or rapport \
statement and false otherwise.\
"""

UNCLEAR_QUESTIONS_JUDGE = """\
You are analyzing a qualitative interview from the participant's perspective.
Read the transcript excerpt below. The interviewer's utterances are numbered [I1], \
[I2], ... in order. Classify each numbered interviewer utterance as either unclear \
or not.

An unclear question is one that is ambiguous, confusing, double-barreled, or \
otherwise likely difficult for a typical participant to understand. When deciding, \
evaluate the question's own wording and structure — not the participant's response. \
A clear question that receives a vague, evasive, or brief answer is still clear; \
participants may give poor answers to perfectly clear questions. Mark a question \
unclear only if a reasonably attentive participant would find it difficult to \
understand or interpret on its own.

Transcript:
{transcript_text}

Return a JSON object with a single key "labels" whose value is a list of booleans \
with one element per numbered interviewer utterance, in the same order ([I1], [I2], \
...). Each element is true if that utterance is unclear and false otherwise.\
"""


# JSON Schemas for constrained decoding (reliable JSON from Gemini 3 thinking
# models). Score judges share the first; classification judges share the second.
_SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
    "additionalProperties": False,
}
_LABELS_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {"type": "array", "items": {"type": "boolean"}},
    },
    "required": ["labels"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_transcript(transcript: list[dict]) -> str:
    lines = []
    for turn in transcript:
        label = "Interviewer" if turn["role"] == "interviewer" else "Participant"
        lines.append(f"{label}: {turn['content']}")
    return "\n".join(lines)


def _format_transcript_numbered(chunk: list[dict]) -> tuple[str, list[str]]:
    """
    Format a chunk with each interviewer utterance numbered [I1], [I2], ... and
    return (formatted_text, interviewer_utterances). The returned list maps a
    label position back to the interviewer utterance it refers to.
    """
    lines = []
    utterances: list[str] = []
    for turn in chunk:
        if turn["role"] == "interviewer":
            utterances.append(turn["content"])
            lines.append(f"[I{len(utterances)}] Interviewer: {turn['content']}")
        else:
            lines.append(f"Participant: {turn['content']}")
    return "\n".join(lines), utterances


def _chunk_transcript(transcript: list[dict], chunk_size: int = 10) -> list[list[dict]]:
    chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
    if len(chunks) >= 2 and len(chunks[-1]) < 5:
        chunks[-2] = chunks[-2] + chunks[-1]
        chunks = chunks[:-1]
    return chunks


def _flagged_utterances(result, utterances: list[str]) -> list[str]:
    """
    Map one chunk's boolean label list onto its interviewer utterances, returning
    the utterances marked true. Defensive against missing/short/long label lists
    (a missing label counts as false; extra labels are ignored).
    """
    raw = result.get("labels", []) if isinstance(result, dict) else []
    if not isinstance(raw, list):
        return []
    return [
        utt for j, utt in enumerate(utterances)
        if j < len(raw) and bool(raw[j])
    ]


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def evaluate_interviewer_behavior(
    transcripts: list[list[dict]],
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    use_batch: bool = False,
) -> dict:
    """
    Evaluate interviewer behavior across five metrics, each a per-participant
    average (mean across the group's per_participant values).

    Each judge runs over 10-turn chunks per transcript.
      - coherence (score 1-4): how logically questions build on each other,
        averaged over chunks then across participants.
      - adaptiveness (score 1-4): how well the interviewer responds to answers,
        averaged over chunks then across participants.
      - leading_questions (avg count/participant): number of interviewer
        utterances classified as leading questions, summed per participant
        (over that participant's chunks) then averaged across participants.
      - support_rapport (avg count/participant): number of interviewer
        utterances classified as containing a support or rapport statement,
        summed per participant (over that participant's chunks) then averaged
        across participants.
      - unclear_questions (avg count/participant): number of interviewer
        utterances classified as unclear, summed per participant (over that
        participant's chunks) then averaged across participants.

    leading_questions, support_rapport, and unclear_questions are derived from a
    per-utterance binary classification. Each participant's "per_participant"
    entry lists the exact utterances flagged, under "leading_utterances",
    "support_rapport_utterances", and "unclear_utterances"; the counts are the
    lengths of those lists.

    Two schemas mean two batches when use_batch=True: the score judges (coherence,
    adaptiveness) and the classification judges (leading, support/rapport,
    unclear) submit separately.

    Args:
        transcripts: List of transcripts, one per participant.
        model, api_base, api_key: Optional LLM overrides.
        use_batch: If True, submit the score judgments and classification labels
            as two batch jobs (one per schema) via LLMClient.batch_call()
            (requires a batch-compatible model).
    """
    if not transcripts:
        warnings.warn(
            "evaluate_interviewer_behavior: transcripts is empty — returning zero values.",
            UserWarning,
            stacklevel=2,
        )
        return {"coherence": 0.0, "adaptiveness": 0.0, "leading_questions": 0.0,
                "support_rapport": 0.0, "unclear_questions": 0.0, "per_participant": []}

    llm = LLMClient(model=model, api_base=api_base, api_key=api_key)
    score_prompts, class_prompts, layout = _build_prompts(transcripts)

    if use_batch:
        score_results = llm.batch_call(score_prompts, json_mode=True, schema=_SCORE_SCHEMA) if score_prompts else []
        class_results = llm.batch_call(class_prompts, json_mode=True, schema=_LABELS_SCHEMA) if class_prompts else []
    else:
        score_results = [llm.call(p, json_mode=True, schema=_SCORE_SCHEMA) for p in score_prompts]
        class_results = [llm.call(p, json_mode=True, schema=_LABELS_SCHEMA) for p in class_prompts]

    return _assemble_results(score_results, class_results, layout)


def _build_prompts(transcripts: list[list[dict]]) -> tuple[list[str], list[str], dict]:
    """
    Build the score and classification prompt lists plus the layout needed to
    reassemble results.

    For each chunk, two score prompts are emitted in order (coherence,
    adaptiveness) and three classification prompts (leading_questions,
    support_rapport, unclear_questions). layout records, per participant, the
    interviewer utterances of each chunk so classification labels can be mapped
    back to utterance text.
    """
    score_prompts: list[str] = []
    class_prompts: list[str] = []
    per_participant: list[dict] = []
    for transcript in transcripts:
        chunks = _chunk_transcript(transcript)
        chunk_utterances: list[list[str]] = []
        for chunk in chunks:
            chunk_text = _format_transcript(chunk)
            score_prompts.append(COHERENCE_JUDGE.format(transcript_text=chunk_text))
            score_prompts.append(ADAPTIVENESS_JUDGE.format(transcript_text=chunk_text))

            numbered_text, utterances = _format_transcript_numbered(chunk)
            class_prompts.append(LEADING_QUESTIONS_JUDGE.format(transcript_text=numbered_text))
            class_prompts.append(SUPPORT_RAPPORT_JUDGE.format(transcript_text=numbered_text))
            class_prompts.append(UNCLEAR_QUESTIONS_JUDGE.format(transcript_text=numbered_text))
            chunk_utterances.append(utterances)
        per_participant.append({"chunk_utterances": chunk_utterances})
    return score_prompts, class_prompts, {"per_participant": per_participant}


def _assemble_results(score_results: list[dict], class_results: list[dict], layout: dict) -> dict:
    # Score results are interleaved two per chunk: [coh_0, adp_0, coh_1, adp_1, ...].
    # Classification results are interleaved three per chunk:
    # [lead_0, supp_0, unclear_0, lead_1, supp_1, unclear_1, ...].
    # Scores are averaged over chunks; classification labels are mapped to the
    # flagged utterances and counted.
    per_participant = []
    score_off = 0
    class_off = 0
    for tl in layout["per_participant"]:
        chunk_utterances = tl["chunk_utterances"]
        n_chunks = len(chunk_utterances)

        score_slice = score_results[score_off:score_off + n_chunks * 2]
        score_off += n_chunks * 2
        class_slice = class_results[class_off:class_off + n_chunks * 3]
        class_off += n_chunks * 3

        # A batch request that failed/returned unparseable output surfaces here as
        # {} (see LLMClient._gemini_batch_retrieve) — average over the chunks that
        # did get scored rather than crashing the whole assembly over one bad
        # request out of many; a chunk missing every score falls back to 1 (the
        # scale's floor) so the average is still defined.
        coherence_scores = [s for s in (score_slice[c * 2].get("score") for c in range(n_chunks)) if s is not None]
        adaptiveness_scores = [s for s in (score_slice[c * 2 + 1].get("score") for c in range(n_chunks)) if s is not None]
        coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1
        adaptiveness = sum(adaptiveness_scores) / len(adaptiveness_scores) if adaptiveness_scores else 1

        leading_utterances: list[str] = []
        support_rapport_utterances: list[str] = []
        unclear_utterances: list[str] = []
        for c in range(n_chunks):
            utterances = chunk_utterances[c]
            leading_utterances.extend(_flagged_utterances(class_slice[c * 3], utterances))
            support_rapport_utterances.extend(_flagged_utterances(class_slice[c * 3 + 1], utterances))
            unclear_utterances.extend(_flagged_utterances(class_slice[c * 3 + 2], utterances))

        per_participant.append({
            "coherence": coherence,
            "adaptiveness": adaptiveness,
            "leading_questions": len(leading_utterances),
            "support_rapport": len(support_rapport_utterances),
            "unclear_questions": len(unclear_utterances),
            "leading_utterances": leading_utterances,
            "support_rapport_utterances": support_rapport_utterances,
            "unclear_utterances": unclear_utterances,
        })

    n = len(per_participant)
    return {
        "coherence": sum(p["coherence"] for p in per_participant) / n,
        "adaptiveness": sum(p["adaptiveness"] for p in per_participant) / n,
        "leading_questions": sum(p["leading_questions"] for p in per_participant) / n,
        "support_rapport": sum(p["support_rapport"] for p in per_participant) / n,
        "unclear_questions": sum(p["unclear_questions"] for p in per_participant) / n,
        "per_participant": per_participant,
    }


def submit_interviewer_behavior(
    transcripts: list[list[dict]],
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Submit an interviewer-behavior batch job and return a JSON-serializable
    handle. Pass the handle to retrieve_interviewer_behavior() later to fetch
    and assemble the results.

    Two batches are submitted because the score judges (coherence, adaptiveness)
    and the classification judges (leading, support/rapport, unclear) use
    different response schemas; the handle records both batch_ids (either may be
    None when there is nothing to score) plus the layout needed to reassemble
    the interleaved results and map classification labels back to utterance
    text. Save it to disk to survive across processes.
    """
    llm = LLMClient(model=model, api_base=api_base, api_key=api_key)
    score_prompts, class_prompts, layout = _build_prompts(transcripts)
    score_batch_id = (
        llm.batch_submit(score_prompts, json_mode=True, schema=_SCORE_SCHEMA)
        if score_prompts else None
    )
    class_batch_id = (
        llm.batch_submit(class_prompts, json_mode=True, schema=_LABELS_SCHEMA)
        if class_prompts else None
    )
    return {
        "function": "interviewer_behavior",
        "score_batch_id": score_batch_id,
        "class_batch_id": class_batch_id,
        "model": llm.model,
        "layout": layout,
    }


def retrieve_interviewer_behavior(
    handle: dict,
    api_key: str | None = None,
) -> dict:
    """
    Retrieve and assemble an interviewer-behavior batch previously submitted
    with submit_interviewer_behavior(). Raises RuntimeError if either batch is
    not yet complete.
    """
    llm = LLMClient(model=handle["model"], api_key=api_key)
    score_results = (
        llm.batch_retrieve(handle["score_batch_id"], json_mode=True)
        if handle.get("score_batch_id") else []
    )
    class_results = (
        llm.batch_retrieve(handle["class_batch_id"], json_mode=True)
        if handle.get("class_batch_id") else []
    )
    return _assemble_results(score_results, class_results, handle["layout"])
