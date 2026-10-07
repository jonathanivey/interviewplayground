"""Tests for the binary-classification behavior of the interviewer_behavior and
participant_experience evaluators.

LLM calls are mocked at the class level. The fake picks a canned response by
inspecting the prompt, so the assertions exercise the real prompt-building and
result-assembly code paths.
"""
import pytest

from interviewplayground.interviewreportcard import (
    evaluate_interviewer_behavior,
    evaluate_participant_experience,
    evaluate_participant_responses,
    evaluate_conversation_length,
)
from interviewplayground.interviewreportcard.interviewer_behavior import (
    _assemble_results as ib_assemble,
    _build_prompts as ib_build,
)
from interviewplayground.interviewreportcard.participant_experience import (
    _assemble_results as pe_assemble,
    _build_prompts as pe_build,
)


def _transcript(interviewer_utts, participant_utts):
    """Interleave interviewer/participant utterances into a transcript."""
    turns = []
    for q, a in zip(interviewer_utts, participant_utts):
        turns.append({"role": "interviewer", "content": q})
        turns.append({"role": "participant", "content": a})
    return turns


# ---------------------------------------------------------------------------
# interviewer_behavior
# ---------------------------------------------------------------------------

def _ib_fake_call(labels_leading, labels_support, labels_unclear=None, coherence=3, adaptiveness=4):
    """Build a side_effect that answers each interviewer-behavior judge prompt."""
    if labels_unclear is None:
        labels_unclear = [False] * len(labels_leading)

    def fake(prompt, json_mode=True, schema=None, **kwargs):
        if "rate the interviewer on coherence" in prompt:
            return {"score": coherence}
        if "rate the interviewer on adaptiveness" in prompt:
            return {"score": adaptiveness}
        if "leading question" in prompt:
            return {"labels": list(labels_leading)}
        if "support or rapport" in prompt:
            return {"labels": list(labels_support)}
        if "unclear" in prompt:
            return {"labels": list(labels_unclear)}
        raise AssertionError(f"unexpected prompt: {prompt[:60]!r}")
    return fake


def test_interviewer_behavior_flags_and_counts(mocker):
    transcript = _transcript(
        ["Don't you think exercise is best?", "Tell me about your week.",
         "You must have been thrilled, right?"],
        ["Sure.", "It was busy.", "I guess."],
    )
    mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        side_effect=_ib_fake_call(labels_leading=[True, False, True],
                                  labels_support=[False, True, False],
                                  labels_unclear=[False, False, True]),
    )

    result = evaluate_interviewer_behavior([transcript])

    assert result["coherence"] == 3.0
    assert result["adaptiveness"] == 4.0
    assert result["leading_questions"] == 2
    assert result["support_rapport"] == 1
    assert result["unclear_questions"] == 1

    pp = result["per_participant"][0]
    assert pp["leading_utterances"] == [
        "Don't you think exercise is best?",
        "You must have been thrilled, right?",
    ]
    assert pp["support_rapport_utterances"] == ["Tell me about your week."]
    assert pp["unclear_utterances"] == ["You must have been thrilled, right?"]
    # counts equal the list lengths.
    assert pp["leading_questions"] == len(pp["leading_utterances"])
    assert pp["support_rapport"] == len(pp["support_rapport_utterances"])
    assert pp["unclear_questions"] == len(pp["unclear_utterances"])


def test_interviewer_behavior_short_label_list_is_defensive(mocker):
    """A label list shorter than the utterance count treats missing labels as false."""
    transcript = _transcript(["Q1?", "Q2?", "Q3?"], ["a", "b", "c"])
    mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        side_effect=_ib_fake_call(labels_leading=[True], labels_support=[], labels_unclear=[]),
    )

    result = evaluate_interviewer_behavior([transcript])
    pp = result["per_participant"][0]
    assert pp["leading_utterances"] == ["Q1?"]
    assert pp["support_rapport_utterances"] == []
    assert pp["unclear_utterances"] == []


def test_interviewer_behavior_multichunk_utterance_mapping():
    """_build_prompts/_assemble_results map labels to the right chunk's utterances."""
    # 20 turns -> two 10-turn chunks, each with 5 interviewer utterances.
    interviewer = [f"Q{i}?" for i in range(10)]
    participant = [f"a{i}" for i in range(10)]
    transcript = _transcript(interviewer, participant)

    score_prompts, class_prompts, layout = ib_build([transcript])
    assert len(score_prompts) == 4  # 2 per chunk
    assert len(class_prompts) == 6  # 3 per chunk
    assert layout["per_participant"][0]["chunk_utterances"] == [
        ["Q0?", "Q1?", "Q2?", "Q3?", "Q4?"],
        ["Q5?", "Q6?", "Q7?", "Q8?", "Q9?"],
    ]

    # interleaved [coh_0, adp_0, coh_1, adp_1] and
    # [lead_0, supp_0, unclear_0, lead_1, supp_1, unclear_1].
    score_results = [{"score": 2}, {"score": 4}, {"score": 4}, {"score": 2}]
    class_results = [
        {"labels": [True, False, False, False, True]},   # chunk 0 leading -> Q0, Q4
        {"labels": [False, True, False, False, False]},  # chunk 0 support -> Q1
        {"labels": [False, False, False, True, False]},  # chunk 0 unclear -> Q3
        {"labels": [False, False, True, False, False]},  # chunk 1 leading -> Q7
        {"labels": [True, False, False, False, False]},  # chunk 1 support -> Q5
        {"labels": [False, False, False, False, True]},  # chunk 1 unclear -> Q9
    ]
    result = ib_assemble(score_results, class_results, layout)
    pp = result["per_participant"][0]
    assert pp["leading_utterances"] == ["Q0?", "Q4?", "Q7?"]
    assert pp["support_rapport_utterances"] == ["Q1?", "Q5?"]
    assert pp["unclear_utterances"] == ["Q3?", "Q9?"]
    assert result["coherence"] == 3.0        # (2 + 4) / 2
    assert result["adaptiveness"] == 3.0     # (4 + 2) / 2


def test_interviewer_behavior_multi_participant_counts_are_averaged(mocker):
    """leading_questions/support_rapport/unclear_questions are averaged across
    participants, not summed."""
    t1 = _transcript(["Q1?"], ["a"])
    t2 = _transcript(["Q2?"], ["b"])
    mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        side_effect=_ib_fake_call(labels_leading=[True], labels_support=[False], labels_unclear=[False]),
    )

    result = evaluate_interviewer_behavior([t1, t2])
    # Both participants have exactly 1 leading question and 0 support/rapport
    # or unclear-question utterances, so the per-participant average equals
    # each participant's own count.
    assert result["leading_questions"] == 1.0
    assert result["support_rapport"] == 0.0
    assert result["unclear_questions"] == 0.0


def test_interviewer_behavior_empty_transcripts_warns():
    with pytest.warns(UserWarning):
        result = evaluate_interviewer_behavior([])
    assert result["leading_questions"] == 0
    assert result["support_rapport"] == 0
    assert result["unclear_questions"] == 0
    assert result["per_participant"] == []


# ---------------------------------------------------------------------------
# participant_experience
# ---------------------------------------------------------------------------

def _pe_fake_call(comfort=3, overall=4):
    def fake(prompt, json_mode=True, schema=None, **kwargs):
        if "comfort level" in prompt:
            return {"score": comfort}
        if "overall quality of the interview" in prompt:
            return {"score": overall}
        raise AssertionError(f"unexpected prompt: {prompt[:60]!r}")
    return fake


def test_participant_experience_scores_and_prompt_count(mocker):
    transcript = _transcript(
        ["How are you and what did you eat?", "Describe your day.",
         "Was it good or bad or neither?"],
        ["Fine.", "It was long.", "Neither."],
    )
    mock = mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        side_effect=_pe_fake_call(comfort=3, overall=5),
    )

    result = evaluate_participant_experience([transcript])

    assert result["comfort_level"] == 3.0
    assert result["overall_experience"] == 5.0

    # comfort and overall are separate prompts: two calls total for one
    # transcript (one chunk -> comfort, overall).
    assert mock.call_count == 2


def test_participant_experience_multi_participant_aggregates(mocker):
    t1 = _transcript(["Q1?", "Q2?"], ["a", "b"])
    t2 = _transcript(["Q3?", "Q4?"], ["c", "d"])
    mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        side_effect=_pe_fake_call(comfort=2, overall=4),
    )

    result = evaluate_participant_experience([t1, t2])
    # comfort and overall are averaged across participants.
    assert result["comfort_level"] == 2.0
    assert result["overall_experience"] == 4.0
    assert len(result["per_participant"]) == 2


def test_participant_experience_chunks_like_interviewer_behavior():
    """comfort/overall judges split transcripts into the exact same chunks as
    the interviewer-behavior judges."""
    interviewer = [f"Q{i}?" for i in range(10)]
    participant = [f"a{i}" for i in range(10)]
    transcript = _transcript(interviewer, participant)  # 20 turns -> two chunks

    _, _, ib_layout = ib_build([transcript])
    pe_score_prompts, pe_layout = pe_build([transcript])

    assert ib_layout["per_participant"][0]["chunk_utterances"] == [
        ["Q0?", "Q1?", "Q2?", "Q3?", "Q4?"], ["Q5?", "Q6?", "Q7?", "Q8?", "Q9?"],
    ]
    assert pe_layout["per_participant"][0]["n_chunks"] == 2
    assert len(pe_score_prompts) == 4  # comfort + overall prompt per chunk

    # comfort/overall are each averaged across the two chunks (3 -> 3.0,
    # 4 -> 4.0 here since both chunks agree, but the averaging itself is what's
    # under test).
    score_results = [{"score": 3}, {"score": 4}, {"score": 3}, {"score": 4}]
    result = pe_assemble(score_results, pe_layout)
    assert result["comfort_level"] == 3.0
    assert result["overall_experience"] == 4.0


def test_participant_experience_empty_transcripts_warns():
    with pytest.warns(UserWarning):
        result = evaluate_participant_experience([])
    assert result["comfort_level"] == 0.0
    assert result["overall_experience"] == 0.0
    assert result["per_participant"] == []


# ---------------------------------------------------------------------------
# participant_responses — RQR judged 1-3, weighted 0 / 1 / 2
# ---------------------------------------------------------------------------

def _rq_fake_call(rqr_score, guide_labels):
    """Answer the RQR relevance judge and the guide-coverage labeler."""
    def fake(prompt, json_mode=True, schema=None, **kwargs):
        if "expert qualitative researcher" in prompt:
            return {"score": rqr_score}
        if "qualitative coding assistant" in prompt:
            return {"labels": guide_labels}
        raise AssertionError(f"unexpected prompt: {prompt[:60]!r}")
    return fake


_GUIDE = [{"topic": "T", "subtopics": ["S1"]}]


def test_participant_responses_top_score_3_weights_fully_and_counts_novel(mocker):
    transcript = _transcript(["Tell me about your care."], ["I saw my doctor last week."])
    mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        # RQR 3 = "directly addresses" (top of 1-3 scale) -> weight 2; no subtopic labeled.
        side_effect=_rq_fake_call(rqr_score=3, guide_labels=[[]]),
    )

    result = evaluate_participant_responses(
        [transcript], research_questions=["RQ1"], interview_guide=_GUIDE,
    )

    wc = len("I saw my doctor last week.".split())  # 5 words
    assert result["relevant_response_volume"] == wc * 2.0
    # max_RQR == 3 and no guide subtopic -> novel.
    assert result["novel_responses"] == 1


def test_participant_responses_mid_score_2_weights_half(mocker):
    transcript = _transcript(["Tell me about your care."], ["I saw my doctor last week."])
    mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        # RQR 2 = "tangentially related" -> weight 1, and not novel.
        side_effect=_rq_fake_call(rqr_score=2, guide_labels=[[]]),
    )

    result = evaluate_participant_responses(
        [transcript], research_questions=["RQ1"], interview_guide=_GUIDE,
    )

    wc = len("I saw my doctor last week.".split())  # 5 words
    assert result["relevant_response_volume"] == wc * 1.0
    assert result["novel_responses"] == 0


def test_participant_responses_score_1_contributes_nothing(mocker):
    transcript = _transcript(["Tell me about your care."], ["The weather is nice today."])
    mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        side_effect=_rq_fake_call(rqr_score=1, guide_labels=[[]]),
    )

    result = evaluate_participant_responses(
        [transcript], research_questions=["RQ1"], interview_guide=_GUIDE,
    )

    # RQR 1 = unrelated: zero weight, and not counted as novel.
    assert result["relevant_response_volume"] == 0
    assert result["novel_responses"] == 0


def test_participant_responses_multi_participant_coverage_is_averaged_not_unioned(mocker):
    """interview_guide_coverage averages each participant's own coverage
    fraction rather than unioning covered subtopics across the whole group."""
    guide = [{"topic": "T", "subtopics": ["S1", "S2"]}]
    t1 = _transcript(["Tell me about your care."], ["I saw my doctor last week."])
    t2 = _transcript(["Tell me about your care."], ["Nothing relevant here."])

    def fake(prompt, json_mode=True, schema=None, **kwargs):
        if "expert qualitative researcher" in prompt:
            return {"score": 3}  # both responses directly relevant
        if "qualitative coding assistant" in prompt:
            if "I saw my doctor last week" in prompt:
                return {"labels": [[1]]}  # covers subtopic 1
            return {"labels": [[]]}  # covers nothing
        raise AssertionError(f"unexpected prompt: {prompt[:60]!r}")

    mocker.patch("interviewplayground.llm_client.LLMClient.call", side_effect=fake)

    result = evaluate_participant_responses(
        [t1, t2], research_questions=["RQ1"], interview_guide=guide,
    )

    # participant 1 covers 1/2 subtopics, participant 2 covers 0/2 -> average 0.25.
    # A union-across-the-group aggregate would instead give 1/2 = 0.5.
    assert result["interview_guide_coverage"] == 0.25
    assert result["per_participant"][0]["interview_guide_coverage"] == 0.5
    assert result["per_participant"][1]["interview_guide_coverage"] == 0.0


# ---------------------------------------------------------------------------
# conversation_length — pure computation from transcripts, no LLM calls
# ---------------------------------------------------------------------------

def test_conversation_length_single_participant():
    transcript = _transcript(["Q1?", "Q2?", "Q3?"], ["one two", "three four five", "six"])
    result = evaluate_conversation_length([transcript])
    assert result["avg_turns"] == 3
    assert result["avg_response_length"] == 6  # 2 + 3 + 1 words
    assert result["per_participant"] == [{"n_turns": 3, "response_length": 6}]


def test_conversation_length_averages_across_participants():
    t1 = _transcript(["Q1?"], ["one two three"])        # 1 turn, 3 words
    t2 = _transcript(["Q1?", "Q2?"], ["a", "b c d e"])   # 2 turns, 5 words

    result = evaluate_conversation_length([t1, t2])
    assert result["avg_turns"] == 1.5
    assert result["avg_response_length"] == 4.0


def test_conversation_length_empty_transcripts_warns():
    with pytest.warns(UserWarning):
        result = evaluate_conversation_length([])
    assert result["avg_turns"] == 0.0
    assert result["avg_response_length"] == 0.0
    assert result["per_participant"] == []
