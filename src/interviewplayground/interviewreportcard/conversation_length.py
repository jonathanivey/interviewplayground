from __future__ import annotations

import warnings


def _participant_turn_stats(transcript: list[dict]) -> tuple[int, int]:
    """Returns (n_turns, response_length) for one transcript: the number of
    participant responses, and their total word count."""
    responses = [t["content"] for t in transcript if t["role"] == "participant"]
    return len(responses), sum(len(r.split()) for r in responses)


def evaluate_conversation_length(transcripts: list[list[dict]]) -> dict:
    """
    Evaluate conversation length across two metrics, each a per-participant
    average (mean across the group's per_participant values). Computed
    directly from the transcripts — no LLM judge calls.

    Metrics:
      - avg_turns: average, per participant, of the number of participant
        responses in that participant's interview.
      - avg_response_length: average, per participant, of that participant's
        total response word count summed across their whole interview.

    Args:
        transcripts: List of transcripts, one per participant. Each transcript
            is a list of {"role": "interviewer"|"participant", "content": str}.
    """
    if not transcripts:
        warnings.warn(
            "evaluate_conversation_length: transcripts is empty — returning zero values.",
            UserWarning,
            stacklevel=2,
        )
        return {"avg_turns": 0.0, "avg_response_length": 0.0, "per_participant": []}

    per_participant = []
    for transcript in transcripts:
        n_turns, response_length = _participant_turn_stats(transcript)
        per_participant.append({"n_turns": n_turns, "response_length": response_length})

    n = len(per_participant)
    return {
        "avg_turns": sum(p["n_turns"] for p in per_participant) / n,
        "avg_response_length": sum(p["response_length"] for p in per_participant) / n,
        "per_participant": per_participant,
    }
