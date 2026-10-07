import pytest
from interviewplayground import Study, load_preset
from interviewplayground.presets import _PRESETS


PRESET_NAMES = sorted(_PRESETS)


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_load_preset_returns_study(name):
    study = load_preset(name)
    assert isinstance(study, Study)


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_preset_has_target_information(name):
    study = load_preset(name)
    assert len(study.target_information) >= 8
    for item in study.target_information:
        assert isinstance(item, str) and item


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_preset_has_participants(name):
    study = load_preset(name)
    assert len(study.participants) >= 1


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_preset_participants_have_memories(name):
    study = load_preset(name)
    for p in study.participants:
        assert len(p.memories) > 0
        non_blank = [m for m in p.memories if not m.is_blank]
        assert len(non_blank) > 0, f"Participant in '{name}' has no filled memories"


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_preset_participants_have_personas(name):
    study = load_preset(name)
    for p in study.participants:
        assert p.persona != "A typical study participant."
        assert len(p.persona) > 20


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_preset_loads_independently(name):
    # Two loads should return independent objects
    a = load_preset(name)
    b = load_preset(name)
    a.participants[0].transcript.append({"role": "test", "content": "x"})
    assert b.participants[0].transcript == []


def test_load_preset_unknown_name():
    with pytest.raises(ValueError, match="Unknown preset"):
        load_preset("nonexistent_study")


def test_all_preset_names_listed_in_error():
    with pytest.raises(ValueError) as exc_info:
        load_preset("bad_name")
    msg = str(exc_info.value)
    for name in PRESET_NAMES:
        assert name in msg
