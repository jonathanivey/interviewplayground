from __future__ import annotations

import warnings

from ..llm_client import LLMClient
# Reuse interviewer_behavior's chunker so comfort/overall judges split
# transcripts into the exact same chunks as the interviewer-behavior judges.
from .interviewer_behavior import _chunk_transcript

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Participant experience is judged by two prompts, both run over 10-turn
# transcript chunks (via _chunk_transcript, split the same way as the
# interviewer-behavior judges) and averaged per participant:
#   - COMFORT_JUDGE returns {"score": <int 1-4>} per chunk.
#   - OVERALL_EXPERIENCE_JUDGE returns {"score": <int 1-5>} per chunk.

COMFORT_JUDGE = """\
You are analyzing a qualitative interview from the participant's perspective.
Read the transcript below and estimate the participant's comfort level during the \
interview.

Transcript:
{transcript_text}

Return a JSON object with a single key "score" whose value is an integer 1-4:
  1 = Entirely uncomfortable or treated unfairly
  2 = Mostly uncomfortable, but with some exceptions
  3 = Mostly comfortable, but with some exceptions
  4 = Entirely comfortable and treated fairly\
"""

OVERALL_EXPERIENCE_JUDGE = """\
You are analyzing a qualitative interview from the participant's perspective.
Read the transcript below and estimate the overall quality of the interview \
experience from the participant's perspective.

Transcript:
{transcript_text}

Return a JSON object with a single key "score" whose value is an integer 1-5:
  1 = Poor
  2 = Below Average
  3 = Average
  4 = Good
  5 = Excellent\
"""


# JSON Schema for constrained decoding (reliable JSON from Gemini 3 thinking
# models). Comfort and overall share this schema.
_SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
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


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def evaluate_participant_experience(
    transcripts: list[list[dict]],
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    use_batch: bool = False,
) -> dict:
    """
    Evaluate participant experience across two metrics, each a per-participant
    average (mean across the group's per_participant values).

    Each judge runs over 10-turn chunks per transcript.
      - comfort_level (score 1-4): comfort rating, averaged over chunks then
        across participants.
      - overall_experience (score 1-5): experience rating, averaged over chunks
        then across participants.

    comfort_level and overall_experience are judged by two separate prompts
    sharing one schema, so use_batch=True submits a single batch job.

    Args:
        transcripts: List of transcripts, one per participant.
        model, api_base, api_key: Optional LLM overrides.
        use_batch: If True, submit the score judgments as a batch job via
            LLMClient.batch_call() (requires a batch-compatible model).
    """
    if not transcripts:
        warnings.warn(
            "evaluate_participant_experience: transcripts is empty — returning zero values.",
            UserWarning,
            stacklevel=2,
        )
        return {"comfort_level": 0.0, "overall_experience": 0.0, "per_participant": []}

    llm = LLMClient(model=model, api_base=api_base, api_key=api_key)
    score_prompts, layout = _build_prompts(transcripts)

    if use_batch:
        score_results = llm.batch_call(score_prompts, json_mode=True, schema=_SCORE_SCHEMA) if score_prompts else []
    else:
        score_results = [llm.call(p, json_mode=True, schema=_SCORE_SCHEMA) for p in score_prompts]

    return _assemble_results(score_results, layout)


def _build_prompts(transcripts: list[list[dict]]) -> tuple[list[str], dict]:
    """
    Build the score prompt list plus the layout needed to reassemble results.

    Per chunk, two score prompts are emitted in order (comfort, overall) —
    chunked identically to the interviewer-behavior judges, via
    _chunk_transcript. layout records each transcript's chunk count.
    """
    score_prompts: list[str] = []
    per_participant: list[dict] = []
    for transcript in transcripts:
        n_chunks = 0
        for chunk in _chunk_transcript(transcript):
            chunk_text = _format_transcript(chunk)
            score_prompts.append(COMFORT_JUDGE.format(transcript_text=chunk_text))
            score_prompts.append(OVERALL_EXPERIENCE_JUDGE.format(transcript_text=chunk_text))
            n_chunks += 1
        per_participant.append({"n_chunks": n_chunks})
    return score_prompts, {"per_participant": per_participant}


def _assemble_results(score_results: list[dict], layout: dict) -> dict:
    # Score results are interleaved two per chunk: [comfort_0, overall_0,
    # comfort_1, overall_1, ...].
    per_participant = []
    score_off = 0
    for tl in layout["per_participant"]:
        n_chunks = tl["n_chunks"]

        score_slice = score_results[score_off:score_off + n_chunks * 2]
        score_off += n_chunks * 2

        # A batch request that failed/returned unparseable output surfaces here as
        # {} (see LLMClient._gemini_batch_retrieve) — .get() drops that chunk from
        # the average rather than crashing the whole assembly over one bad request
        # out of many; a participant whose chunks all failed gets None (excluded
        # from the aggregate mean below) rather than a faked score.
        comfort_scores = [s for s in (score_slice[c * 2].get("score") for c in range(n_chunks)) if s is not None]
        overall_scores = [s for s in (score_slice[c * 2 + 1].get("score") for c in range(n_chunks)) if s is not None]

        per_participant.append({
            "comfort_level": sum(comfort_scores) / len(comfort_scores) if comfort_scores else None,
            "overall_experience": sum(overall_scores) / len(overall_scores) if overall_scores else None,
        })

    def _mean(key: str) -> float:
        values = [p[key] for p in per_participant if p[key] is not None]
        return sum(values) / len(values) if values else 0.0

    return {
        "comfort_level": _mean("comfort_level"),
        "overall_experience": _mean("overall_experience"),
        "per_participant": per_participant,
    }


def submit_participant_experience(
    transcripts: list[list[dict]],
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Submit a participant-experience batch job and return a JSON-serializable
    handle. Pass the handle to retrieve_participant_experience() later (e.g. the
    next day) to fetch and assemble the results.

    Save it to disk (json.dump) to survive across processes.
    """
    llm = LLMClient(model=model, api_base=api_base, api_key=api_key)
    score_prompts, layout = _build_prompts(transcripts)
    batch_id = (
        llm.batch_submit(score_prompts, json_mode=True, schema=_SCORE_SCHEMA)
        if score_prompts else None
    )
    return {
        "function": "participant_experience",
        "batch_id": batch_id,
        "model": llm.model,
        "layout": layout,
    }


def retrieve_participant_experience(
    handle: dict,
    api_key: str | None = None,
) -> dict:
    """
    Retrieve and assemble a participant-experience batch previously submitted
    with submit_participant_experience(). Raises RuntimeError if the batch is
    not yet complete.
    """
    llm = LLMClient(model=handle["model"], api_key=api_key)
    score_results = (
        llm.batch_retrieve(handle["batch_id"], json_mode=True)
        if handle.get("batch_id") else []
    )
    return _assemble_results(score_results, handle["layout"])
