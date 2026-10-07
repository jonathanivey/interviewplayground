import pytest
from interviewplayground import Study, Participant


def _make_study(n_targets: int = 5, n_participants: int = 3, memories_each: int = 8) -> Study:
    study = Study(target_information=[f"Target {i}" for i in range(n_targets)])
    study.create_participants(n_participants)
    for p in study.participants:
        p.create_blank_memories(memories_each)
    return study


def test_n_property():
    study = Study(target_information=["A", "B"])
    assert study.n == 0
    study.create_participants(3)
    assert study.n == 3


def test_create_participants_defaults():
    study = Study(target_information=["A"])
    study.create_participants(2)
    for p in study.participants:
        assert isinstance(p, Participant)
        assert p.knowledge == "Medium"
        assert p.memories == []


def test_distribute_assigns_target_indices():
    study = _make_study(n_targets=5, n_participants=3, memories_each=8)
    study.distribute_target_information(avg_per_participant=3.0)

    for p in study.participants:
        target_mems = [m for m in p.memories if m.is_target]
        assert len(target_mems) >= 1
        for m in target_mems:
            assert len(m.target_indices) == 1
            assert 0 <= m.target_indices[0] < 5


def test_distribute_ensure_all_covered():
    study = _make_study(n_targets=10, n_participants=3, memories_each=10)
    study.distribute_target_information(avg_per_participant=4.0, ensure_all_distributed=True)

    all_covered = set()
    for p in study.participants:
        for m in p.memories:
            all_covered.update(m.target_indices)

    assert all_covered == set(range(10))


def test_distribute_not_ensure_all_covered_may_miss():
    # With very few avg_per_participant and many targets it's unlikely but possible
    # to miss some. We just verify the flag is respected when False — no assertion
    # that coverage is incomplete (it's probabilistic), but we confirm no error.
    study = _make_study(n_targets=5, n_participants=3, memories_each=8)
    study.distribute_target_information(avg_per_participant=2.0, ensure_all_distributed=False)
    # Should complete without error


def test_distribute_raises_if_not_enough_blanks():
    study = Study(target_information=[f"T{i}" for i in range(10)])
    study.create_participants(1)
    study.participants[0].create_blank_memories(2)  # only 2 blanks but needs ~5

    with pytest.raises(ValueError, match="blank memories"):
        study.distribute_target_information(avg_per_participant=5.0)


def test_distribute_nontarget_slots_remain_blank():
    study = _make_study(n_targets=3, n_participants=2, memories_each=6)
    study.distribute_target_information(avg_per_participant=2.0)

    for p in study.participants:
        non_target = [m for m in p.memories if not m.is_target]
        for m in non_target:
            assert m.is_blank


def test_evaluate_defaults_interview_guide_from_self(mocker):
    mocker.patch("interviewplayground.study.evaluate_conversation_length", return_value={})
    mock_rq = mocker.patch("interviewplayground.study.evaluate_participant_responses", return_value={})
    mocker.patch("interviewplayground.study.evaluate_interviewer_behavior", return_value={})
    mocker.patch("interviewplayground.study.evaluate_participant_experience", return_value={})

    study = Study(target_information=["A"])
    study.interview_guide = [{"topic": "T", "subtopics": ["S"]}]
    study.create_participants(1)
    study.participants[0].transcript = [{"role": "interviewer", "content": "Hi"}]

    study.evaluate()

    assert mock_rq.call_args.kwargs["interview_guide"] == study.interview_guide


def test_from_description_calls_llm(mocker):
    mock_call = mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        return_value={"target_information": ["Topic A", "Topic B", "Topic C"]},
    )

    study = Study.from_description("A study about something.")

    mock_call.assert_called_once()
    assert study.target_information == ["Topic A", "Topic B", "Topic C"]
    assert study.n == 0
