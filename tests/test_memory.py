from interviewplayground import Memory


def test_defaults():
    m = Memory()
    assert m.content == ""
    assert m.target_indices == []
    assert m.reflexive is False
    assert m.sensitive is False


def test_is_blank():
    m = Memory()
    assert m.is_blank
    m.content = "something"
    assert not m.is_blank


def test_is_target():
    m = Memory()
    assert not m.is_target
    m.target_indices = [0]
    assert m.is_target


def test_mutable_defaults_are_independent():
    a = Memory()
    b = Memory()
    a.target_indices.append(1)
    assert b.target_indices == []
