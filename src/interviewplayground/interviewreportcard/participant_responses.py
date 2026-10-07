from __future__ import annotations

import warnings

from ..llm_client import LLMClient

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Rates how relevant a participant response is to a given research question.
# Returns {"score": <int 1-3>}
RQR_JUDGE = """
You are an expert qualitative researcher analyzing interview data.

Estimate how relevant the participant statement from the current interview excerpt below is to the provided research question on a scale from 1 to 3. In addition to the current excerpt and research question, you are also provided with a short context blurb and the interview excerpt that immediately preceded the current excerpt in the transcript. These two sections are only to understand the context of the current excerpt, and your rating should be for participant statement in the current excerpt.

Scoring Rubric:
1. The participant statement is unrelated to the research question or discusses a completely different topic.
2. The participant statement is tangentially related to the topic of the research question.
3. The participant statement directly addresses the research question.

RESEARCH QUESTION:
{research_question}

PREVIOUS INTERVIEW EXCERPT (context_only):
{previous_excerpt}

CURRENT INTERVIEW EXCERPT (rate this):
{current_excerpt}


Return a JSON object with a single key "score" whose value is an integer (1, 2, or 3).

"""

# Labels each participant response in a transcript with the guide subtopic
# indices (1-based) it addresses. A response may address several subtopics.
# Returns {"labels": [[<int>, ...], ...]}  (one inner list per excerpt)
GUIDE_LABEL = """\
You are a qualitative coding assistant. Below is a numbered list of interview \
guide subtopics followed by a list of interview excerpts from a single interview. \
Each excerpt shows the interviewer's question and the participant's response.

For each excerpt, identify every subtopic that the participant's response \
directly addresses. A single response may address multiple subtopics, one \
subtopic, or none. Base your labels on what the participant's response actually \
says — not on what the interviewer's question asked about. If the interviewer \
raises a subtopic but the participant does not engage with it, do not label it. \
If the response does not clearly address any subtopic, use an empty list.

Interview guide subtopics:
{subtopics_text}

Interview excerpts:
{excerpts_json}

Return a JSON object with a single key "labels" whose value is a list with one \
element per excerpt, in the same order as the excerpts. Each element is itself a \
list of the subtopic numbers (integers) that the participant's response in that \
excerpt directly addresses, or an empty list [] if it addresses none.\
"""


# The judge rates on the 1-3 rubric above; the metrics use these weights instead,
# so an unrelated response contributes nothing and a directly-relevant one counts
# fully. Anything outside 1-3 (a judge ignoring the rubric) gets weight 0.
_RQR_WEIGHT = {1: 0.0, 2: 1.0, 3: 2.0}


# JSON Schemas for constrained decoding (reliable JSON from Gemini 3 thinking models).
_RQR_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
    "additionalProperties": False,
}
_LABELS_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        }
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


def _flatten_subtopics(interview_guide: list[dict]) -> list[str]:
    subtopics = []
    for topic in interview_guide:
        topic_name = topic.get("topic", "")
        for sub in topic.get("subtopics", []):
            subtopics.append(f"{topic_name}: {sub}" if topic_name else sub)
    return subtopics


def _format_subtopics(subtopics: list[str]) -> str:
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(subtopics))


def _format_responses(responses: list[str]) -> str:
    import json
    return json.dumps(responses, ensure_ascii=False, indent=2)


def _clean_label_list(item, total_subtopics: int) -> list[int]:
    """
    Normalize one excerpt's raw labels into a sorted, deduped list of valid
    1-based subtopic indices. Accepts a list (expected), a bare int, or null
    (defensive against schema-less / older responses).
    """
    if isinstance(item, int):
        item = [item]
    elif not isinstance(item, list):
        return []
    return sorted({
        lbl for lbl in item
        if isinstance(lbl, int) and 1 <= lbl <= total_subtopics
    })


def _extract_pairs(transcript: list[dict]) -> list[tuple[str | None, str]]:
    """Extract (interviewer_question, participant_response) pairs in order."""
    pairs: list[tuple[str | None, str]] = []
    last_interviewer: str | None = None
    for turn in transcript:
        if turn["role"] == "interviewer":
            last_interviewer = turn["content"]
        elif turn["role"] == "participant":
            pairs.append((last_interviewer, turn["content"]))
    return pairs


def _build_excerpts(pairs: list[tuple[str | None, str]]) -> list[str]:
    return [
        f"Interviewer: {q}\nParticipant: {r}" if q else f"Participant: {r}"
        for q, r in pairs
    ]


def _excerpt_chunks(excerpts: list[str]) -> list[list[str]]:
    """Chunk excerpts in groups of 10, merging a small trailing chunk (<5) back."""
    chunks = [excerpts[i:i + 10] for i in range(0, len(excerpts), 10)]
    if len(chunks) >= 2 and len(chunks[-1]) < 5:
        chunks[-2] = chunks[-2] + chunks[-1]
        chunks = chunks[:-1]
    return chunks


def _labels_from_results(
    chunk_results: list[dict], chunk_sizes: list[int], total_items: int
) -> list[list[int]]:
    """Reconstruct one cleaned label-list per excerpt from chunked label results."""
    labels: list[list[int]] = []
    for result, size in zip(chunk_results, chunk_sizes):
        raw = result.get("labels", []) if isinstance(result, dict) else []
        for j in range(size):
            item = raw[j] if j < len(raw) else []
            labels.append(_clean_label_list(item, total_items))
    return labels


def _build_prompts(
    transcripts: list[list[dict]],
    research_questions: list[str],
    subtopics_text: str,
    total_subtopics: int,
) -> tuple[list[str], list[str], dict]:
    """
    Build every participant-responses prompt plus the layout needed to
    reassemble results. Returns (rqr_prompts, label_prompts, layout).

    rqr_prompts are ordered response-major, research-question-minor. label_prompts
    are grouped per transcript as guide chunks and share _LABELS_SCHEMA. layout
    carries the per-transcript response word counts and chunk sizes
    _assemble() needs.
    """
    rqr_prompts: list[str] = []
    label_prompts: list[str] = []
    per_transcript: list[dict] = []

    for transcript in transcripts:
        pairs = _extract_pairs(transcript)
        excerpts = _build_excerpts(pairs)
        word_counts = [len(r.split()) for _, r in pairs]

        for i in range(len(pairs)):
            current = excerpts[i]
            previous = excerpts[i - 1] if i > 0 else "N/A"
            for rq in research_questions:
                rqr_prompts.append(RQR_JUDGE.format(
                    research_question=rq,
                    current_excerpt=current,
                    previous_excerpt=previous,
                ))

        guide_sizes: list[int] = []
        if excerpts and total_subtopics:
            for chunk in _excerpt_chunks(excerpts):
                label_prompts.append(GUIDE_LABEL.format(
                    subtopics_text=subtopics_text,
                    excerpts_json=_format_responses(chunk),
                ))
                guide_sizes.append(len(chunk))

        per_transcript.append({
            "word_counts": word_counts,
            "guide_sizes": guide_sizes,
        })

    layout = {
        "n_rq": len(research_questions),
        "total_subtopics": total_subtopics,
        "per_transcript": per_transcript,
    }
    return rqr_prompts, label_prompts, layout


def _assemble_results(
    rqr_results: list[dict], label_results: list[dict], layout: dict
) -> dict:
    """Reassemble the participant-responses metrics from flat batch/sequential
    results.

    All three top-level metrics are per-participant averages (mean across the
    group's per_participant values), not sums or group-wide unions.
    """
    n_rq = layout["n_rq"]
    total_subtopics = layout["total_subtopics"]

    per_participant = []

    rqr_off = 0
    label_off = 0
    for tl in layout["per_transcript"]:
        word_counts = tl["word_counts"]
        n_resp = len(word_counts)

        rqr_slice = rqr_results[rqr_off:rqr_off + n_resp * n_rq]
        rqr_off += n_resp * n_rq

        n_guide = len(tl["guide_sizes"])
        guide_slice = label_results[label_off:label_off + n_guide]
        label_off += n_guide

        guide_labels = _labels_from_results(guide_slice, tl["guide_sizes"], total_subtopics)
        if not guide_labels:
            guide_labels = [[] for _ in range(n_resp)]

        p_rqr_sum = 0.0
        max_rqr_per_response: list[int] = []
        for i in range(n_resp):
            # A batch request that failed/returned unparseable output surfaces here
            # as {} (see LLMClient._gemini_batch_retrieve) — .get() skips it rather
            # than crashing the whole assembly over one bad request out of many.
            raw_scores = [rqr_slice[i * n_rq + j].get("score") for j in range(n_rq)]
            scores = [s for s in raw_scores if isinstance(s, int)]
            max_rqr = max(scores) if scores else 1  # 1 = floor of the 1-3 RQR scale
            max_rqr_per_response.append(max_rqr)
            weight = _RQR_WEIGHT.get(max_rqr, 0.0)
            p_rqr_sum += word_counts[i] * weight

        p_covered: set[int] = set()
        p_novel = 0
        for max_rqr, label_list in zip(max_rqr_per_response, guide_labels):
            p_covered.update(label_list)
            if max_rqr == 3 and not label_list:  # 3 = top of the 1-3 RQR scale ("directly addresses")
                p_novel += 1

        per_participant.append({
            "relevant_response_volume": p_rqr_sum,
            "interview_guide_coverage": len(p_covered) / total_subtopics if total_subtopics else 0.0,
            "novel_responses": p_novel,
            "subtopic_labels": guide_labels,
        })

    if not per_participant:
        return {**_EMPTY_RESULT, "per_participant": []}

    n = len(per_participant)
    return {
        "relevant_response_volume": sum(p["relevant_response_volume"] for p in per_participant) / n,
        "interview_guide_coverage": sum(p["interview_guide_coverage"] for p in per_participant) / n,
        "novel_responses": sum(p["novel_responses"] for p in per_participant) / n,
        "per_participant": per_participant,
    }


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

_EMPTY_RESULT = {
    "relevant_response_volume": 0.0,
    "interview_guide_coverage": 0.0,
    "novel_responses": 0.0,
    "per_participant": [],
}


def _prepare(interview_guide):
    """Resolve subtopics shared by evaluate/submit. Returns the text and
    counts needed to build prompts."""
    subtopics = _flatten_subtopics(interview_guide)
    return {
        "subtopics_text": _format_subtopics(subtopics),
        "total_subtopics": len(subtopics),
    }


def evaluate_participant_responses(
    transcripts: list[list[dict]],
    research_questions: list[str],
    interview_guide: list[dict],
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    use_batch: bool = False,
) -> dict:
    """
    Evaluate participant responses across three metrics. Each is a
    per-participant average (mean across the group's per_participant values)
    — see the "per_participant" list for each participant's own value.

    Metrics:
      - relevant_response_volume: average, per participant, of
        sum(word_count * RQR_weight) across that participant's responses. RQR
        is scored 1-3 by an LLM judge per (response, research_question) pair;
        the max over all questions is taken and converted to a weight of 0, 1,
        or 2 (scores 1, 2, 3). Unrelated content (RQR 1) contributes nothing.
      - interview_guide_coverage: average, per participant, of that
        participant's own fraction of guide subtopics addressed by at least
        one of their responses.
      - novel_responses: average, per participant, of that participant's count
        of responses where max_RQR == 3 and no guide subtopic was assigned.

    Args:
        transcripts: List of transcripts, one per participant. Each transcript
            is a list of {"role": "interviewer"|"participant", "content": str}.
        research_questions: List of research question strings.
        interview_guide: List of topic dicts with "topic" and "subtopics" keys.
        model, api_base, api_key: Optional LLM overrides.
        use_batch: If True, submit the relevance judgments and coverage labels
            as two batch jobs (one per schema) via LLMClient.batch_call()
            (requires a batch-compatible model).
    """
    for name, value in (("transcripts", transcripts),
                        ("research_questions", research_questions),
                        ("interview_guide", interview_guide)):
        if not value:
            warnings.warn(
                f"evaluate_participant_responses: {name} is empty — returning zero values.",
                UserWarning,
                stacklevel=2,
            )
            return {**_EMPTY_RESULT, "per_participant": []}

    llm = LLMClient(model=model, api_base=api_base, api_key=api_key)
    prep = _prepare(interview_guide)
    rqr_prompts, label_prompts, layout = _build_prompts(
        transcripts, research_questions, prep["subtopics_text"], prep["total_subtopics"],
    )

    if use_batch:
        rqr_results = llm.batch_call(rqr_prompts, json_mode=True, schema=_RQR_SCHEMA) if rqr_prompts else []
        label_results = llm.batch_call(label_prompts, json_mode=True, schema=_LABELS_SCHEMA) if label_prompts else []
    else:
        rqr_results = [llm.call(p, json_mode=True, schema=_RQR_SCHEMA) for p in rqr_prompts]
        label_results = [llm.call(p, json_mode=True, schema=_LABELS_SCHEMA) for p in label_prompts]

    return _assemble_results(rqr_results, label_results, layout)


def submit_participant_responses(
    transcripts: list[list[dict]],
    research_questions: list[str],
    interview_guide: list[dict],
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Submit participant-responses evaluation as batch jobs and return a
    JSON-serializable handle. Pass it to retrieve_participant_responses()
    later to fetch and assemble the results.

    Two batches are submitted because the relevance judge and the coverage
    labeler use different response schemas; the handle records both batch_ids
    (either may be None when there is nothing to score) plus the layout needed
    to reassemble the interleaved results. Save it to disk to survive across
    processes.
    """
    llm = LLMClient(model=model, api_base=api_base, api_key=api_key)
    prep = _prepare(interview_guide)
    rqr_prompts, label_prompts, layout = _build_prompts(
        transcripts, research_questions, prep["subtopics_text"], prep["total_subtopics"],
    )
    rqr_batch_id = (
        llm.batch_submit(rqr_prompts, json_mode=True, schema=_RQR_SCHEMA)
        if rqr_prompts else None
    )
    label_batch_id = (
        llm.batch_submit(label_prompts, json_mode=True, schema=_LABELS_SCHEMA)
        if label_prompts else None
    )
    return {
        "function": "participant_responses",
        "rqr_batch_id": rqr_batch_id,
        "label_batch_id": label_batch_id,
        "model": llm.model,
        "layout": layout,
    }


def retrieve_participant_responses(
    handle: dict,
    api_key: str | None = None,
) -> dict:
    """
    Retrieve and assemble a participant-responses evaluation previously
    submitted with submit_participant_responses(). Raises RuntimeError if
    either batch is not yet complete.
    """
    llm = LLMClient(model=handle["model"], api_key=api_key)
    rqr_results = (
        llm.batch_retrieve(handle["rqr_batch_id"], json_mode=True)
        if handle.get("rqr_batch_id") else []
    )
    label_results = (
        llm.batch_retrieve(handle["label_batch_id"], json_mode=True)
        if handle.get("label_batch_id") else []
    )
    return _assemble_results(rqr_results, label_results, handle["layout"])
