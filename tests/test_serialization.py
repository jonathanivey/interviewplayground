import json
import pathlib
import tempfile

import pytest

from interviewplayground import Memory, Participant, Study


# ── Memory ──────────────────────────────────────────────────────────────────

def test_memory_to_dict_roundtrip():
    m = Memory(content="A vivid memory.", insight_indices=[2], reflexive=True, sensitive=False)
    assert Memory.from_dict(m.to_dict()).content == m.content
    assert Memory.from_dict(m.to_dict()).insight_indices == [2]
    assert Memory.from_dict(m.to_dict()).reflexive is True


def test_memory_from_dict_blank():
    m = Memory.from_dict({"content": "", "insight_indices": [], "reflexive": False, "sensitive": False})
    assert m.is_blank
    assert not m.is_insight


# ── Participant ──────────────────────────────────────────────────────────────

def _sample_participant() -> Participant:
    p = Participant()
    p.persona = "A test participant."
    p.knowledge = "High"
    p.verbosity = "Low"
    p.disclosure = "High"
    p.transcript = [{"role": "interviewer", "content": "Hello."}]
    m = Memory(content="I remember something.", insight_indices=[0], reflexive=True, sensitive=False)
    p.memories = [m, Memory()]
    return p


def test_participant_to_dict_roundtrip():
    p = _sample_participant()
    d = p.to_dict()
    p2 = Participant.from_dict(d)

    assert p2.persona == p.persona
    assert p2.knowledge == "High"
    assert p2.verbosity == "Low"
    assert p2.memory == "Medium"
    assert p2.disclosure == "High"
    assert len(p2.memories) == 2
    assert p2.memories[0].content == "I remember something."
    assert p2.memories[0].insight_indices == [0]
    assert p2.memories[0].reflexive is True
    assert p2.memories[1].is_blank
    assert p2.transcript == [{"role": "interviewer", "content": "Hello."}]


def test_participant_save_load(tmp_path):
    p = _sample_participant()
    filepath = str(tmp_path / "participant.json")
    p.save(filepath)

    p2 = Participant.load(filepath)
    assert p2.persona == p.persona
    assert len(p2.memories) == len(p.memories)
    assert p2.memories[0].content == p.memories[0].content
    assert p2.transcript == p.transcript


def test_participant_save_is_valid_json(tmp_path):
    p = _sample_participant()
    filepath = tmp_path / "participant.json"
    p.save(str(filepath))
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    assert "persona" in data
    assert "memories" in data
    assert isinstance(data["memories"], list)


def test_participant_load_missing_file():
    with pytest.raises(FileNotFoundError):
        Participant.load("/nonexistent/path/participant.json")


# ── Study ────────────────────────────────────────────────────────────────────

def _sample_study() -> Study:
    study = Study(insights=["Topic A", "Topic B"])
    study.create_participants(2)
    study.participants[0].persona = "First participant."
    study.participants[0].create_blank_memories(3)
    study.participants[0].memories[0].content = "A filled memory."
    study.participants[0].memories[0].insight_indices = [0]
    study.participants[1].persona = "Second participant."
    return study


def test_study_save_load(tmp_path):
    study = _sample_study()
    filepath = str(tmp_path / "study.json")
    study.save(filepath)

    study2 = Study.load(filepath)
    assert study2.insights == ["Topic A", "Topic B"]
    assert study2.n == 2
    assert study2.participants[0].persona == "First participant."
    assert study2.participants[0].memories[0].content == "A filled memory."
    assert study2.participants[0].memories[0].insight_indices == [0]
    assert study2.participants[1].persona == "Second participant."


def test_study_save_load_roundtrip_preserves_interview_guide(tmp_path):
    study = _sample_study()
    study.interview_guide = [{"topic": "Topic A", "subtopics": ["Sub 1", "Sub 2"]}]
    filepath = str(tmp_path / "study.json")
    study.save(filepath)

    study2 = Study.load(filepath)
    assert study2.interview_guide == study.interview_guide


def test_study_load_defaults_interview_guide_when_absent(tmp_path):
    filepath = tmp_path / "study.json"
    filepath.write_text(json.dumps({
        "insights": ["A"], "research_questions": [], "participants": [],
    }))

    study = Study.load(str(filepath))
    assert study.interview_guide == []


def test_study_save_is_valid_json(tmp_path):
    study = _sample_study()
    filepath = tmp_path / "study.json"
    study.save(str(filepath))
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    assert "insights" in data
    assert "participants" in data
    assert len(data["participants"]) == 2


def test_study_save_load_roundtrip_preserves_transcript(tmp_path):
    study = _sample_study()
    study.participants[0].transcript = [
        {"role": "interviewer", "content": "Q1"},
        {"role": "participant", "content": "A1"},
    ]
    filepath = str(tmp_path / "study.json")
    study.save(filepath)

    study2 = Study.load(filepath)
    assert study2.participants[0].transcript == study.participants[0].transcript


def test_study_load_missing_file():
    with pytest.raises(FileNotFoundError):
        Study.load("/nonexistent/path/study.json")


def test_study_load_independent_of_original(tmp_path):
    study = _sample_study()
    filepath = str(tmp_path / "study.json")
    study.save(filepath)

    study2 = Study.load(filepath)
    study2.participants[0].transcript.append({"role": "test", "content": "x"})
    assert study.participants[0].transcript == []
