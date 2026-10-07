import pytest
from interviewplayground import Memory, Participant


def test_defaults():
    p = Participant()
    assert p.persona == "A typical study participant."
    assert p.memories == []
    assert p.knowledge == "Medium"
    assert p.verbosity == "Medium"
    assert p.memory == "Medium"
    assert p.reflexivity == "Medium"
    assert p.disclosure == "Medium"
    assert p.understanding == "Medium"
    assert p.transcript == []


def test_memory_size_property():
    p = Participant()
    assert p.memory_size == 0
    p.create_blank_memories(5)
    assert p.memory_size == 5


def test_create_blank_memories():
    p = Participant()
    p.create_blank_memories(3)
    assert len(p.memories) == 3
    for m in p.memories:
        assert isinstance(m, Memory)
        assert m.is_blank
        assert not m.is_target


def test_create_blank_memories_appends():
    p = Participant()
    p.create_blank_memories(2)
    p.create_blank_memories(3)
    assert len(p.memories) == 5


def test_generate_nontarget_memories(mocker):
    p = Participant()
    p.create_blank_memories(2)

    fake_result = {
        "memories": [
            {"content": "Memory A", "reflexive": True, "sensitive": False},
            {"content": "Memory B", "reflexive": False, "sensitive": True},
        ]
    }
    mock_call = mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        return_value=fake_result,
    )

    p.generate_nontarget_memories()

    mock_call.assert_called_once()
    assert p.memories[0].content == "Memory A"
    assert p.memories[0].reflexive is True
    assert p.memories[0].sensitive is False
    assert p.memories[1].content == "Memory B"
    assert p.memories[1].sensitive is True


def test_generate_nontarget_memories_skips_target_slots(mocker):
    p = Participant()
    p.create_blank_memories(3)
    p.memories[0].target_indices = [0]  # mark one as target

    fake_result = {
        "memories": [
            {"content": "Non-target A", "reflexive": False, "sensitive": False},
            {"content": "Non-target B", "reflexive": False, "sensitive": False},
        ]
    }
    mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        return_value=fake_result,
    )

    p.generate_nontarget_memories()

    # The target memory should still be blank
    assert p.memories[0].is_blank
    # The other two should be filled
    non_target_contents = {m.content for m in p.memories if not m.is_target}
    assert non_target_contents == {"Non-target A", "Non-target B"}


def test_generate_target_memories(mocker):
    p = Participant()
    p.create_blank_memories(2)
    p.memories[0].target_indices = [0]
    p.memories[1].target_indices = [1]

    fake_result = {
        "memories": [
            {"content": "Target mem 0", "reflexive": False, "sensitive": False},
            {"content": "Target mem 1", "reflexive": True, "sensitive": True},
        ]
    }
    mock_call = mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        return_value=fake_result,
    )

    target_info = ["Topic A", "Topic B"]
    p.generate_target_memories(target_info)

    mock_call.assert_called_once()
    target_mems = [m for m in p.memories if m.is_target]
    assert target_mems[0].content == "Target mem 0"
    assert target_mems[1].content == "Target mem 1"
    assert target_mems[1].reflexive is True
    assert target_mems[1].sensitive is True


def test_ask_appends_to_transcript(mocker):
    p = Participant()
    p.create_blank_memories(1)
    p.memories[0].content = "I remember something."

    mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        return_value="That's a great question.",
    )
    mocker.patch(
        "interviewplayground.llm_client.LLMClient.embed",
        side_effect=[[[1.0, 0.0]], [[1.0, 0.0]]],  # memory embedding, then question embedding
    )

    response = p.ask("Tell me about yourself.")

    assert response == "That's a great question."
    assert len(p.transcript) == 2
    assert p.transcript[0] == {"role": "interviewer", "content": "Tell me about yourself."}
    assert p.transcript[1] == {"role": "participant", "content": "That's a great question."}


def test_ask_builds_transcript_context(mocker):
    p = Participant()  # no memories — embed never called

    mock_call = mocker.patch(
        "interviewplayground.llm_client.LLMClient.call",
        return_value="Response.",
    )

    p.ask("First question.")
    p.ask("Second question.")

    # Second call's prompt should include the first exchange
    second_prompt = mock_call.call_args_list[1][0][0]
    assert "First question." in second_prompt
    assert "Response." in second_prompt


# ── Retrieval ────────────────────────────────────────────────────────────────

def test_retrieve_ranks_by_cosine_similarity(mocker):
    p = Participant()
    p.retrieval_top_k = 2
    for content in ["Memory A", "Memory B", "Memory C"]:
        m = Memory(content=content)
        p.memories.append(m)

    # Memory A: [1, 0]  — parallel to question → similarity 1.0
    # Memory B: [0, 1]  — perpendicular       → similarity 0.0
    # Memory C: [1, 1]  — 45°                 → similarity 0.707
    # Question: [1, 0]
    mock_embed = mocker.patch("interviewplayground.llm_client.LLMClient.embed")
    mock_embed.side_effect = [
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],  # memory embeddings (batched)
        [[1.0, 0.0]],                            # question embedding
    ]

    retrieved = p._retrieve_relevant_memories("some question")

    assert len(retrieved) == 2
    assert retrieved[0].content == "Memory A"  # highest similarity
    assert retrieved[1].content == "Memory C"  # second highest


def test_retrieve_returns_all_when_fewer_than_top_k(mocker):
    p = Participant()
    p.retrieval_top_k = 10
    p.memories.append(Memory(content="Only memory."))

    mocker.patch(
        "interviewplayground.llm_client.LLMClient.embed",
        side_effect=[[[1.0, 0.0]], [[1.0, 0.0]]],
    )

    retrieved = p._retrieve_relevant_memories("question")
    assert len(retrieved) == 1


def test_retrieve_returns_empty_when_no_memories(mocker):
    p = Participant()
    mock_embed = mocker.patch("interviewplayground.llm_client.LLMClient.embed")

    retrieved = p._retrieve_relevant_memories("question")

    assert retrieved == []
    mock_embed.assert_not_called()


def test_retrieve_caches_memory_embeddings(mocker):
    p = Participant()
    for content in ["Memory X", "Memory Y"]:
        p.memories.append(Memory(content=content))

    mock_embed = mocker.patch("interviewplayground.llm_client.LLMClient.embed")
    mock_embed.side_effect = [
        [[1.0, 0.0], [0.0, 1.0]],  # first ask: memory embeddings (2 texts)
        [[1.0, 0.0]],               # first ask: question embedding
        [[0.5, 0.5]],               # second ask: question only (memories cached)
    ]

    p._retrieve_relevant_memories("first question")
    p._retrieve_relevant_memories("second question")

    assert mock_embed.call_count == 3
    # Third call should only embed the question, not the memories
    third_call_texts = mock_embed.call_args_list[2][0][0]
    assert third_call_texts == ["second question"]


def test_retrieve_skips_blank_memories(mocker):
    p = Participant()
    p.memories.append(Memory(content="Filled memory."))
    p.memories.append(Memory())  # blank — should be excluded

    mock_embed = mocker.patch("interviewplayground.llm_client.LLMClient.embed")
    mock_embed.side_effect = [
        [[1.0, 0.0]],  # only one memory embedded
        [[1.0, 0.0]],
    ]

    retrieved = p._retrieve_relevant_memories("question")

    assert len(retrieved) == 1
    assert retrieved[0].content == "Filled memory."
    # First embed call should only contain the filled memory's content
    first_call_texts = mock_embed.call_args_list[0][0][0]
    assert first_call_texts == ["Filled memory."]
