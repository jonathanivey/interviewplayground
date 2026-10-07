import pytest

from interviewplayground.speaking_time import (
    SPEECH_DELAY_BY_VERBOSITY,
    SPEECH_RATE_SPM_BY_VERBOSITY,
    count_syllables,
    estimate_speaking_duration,
)

SAMPLE_TEXT = "Well, I guess it really depends on how you look at the whole situation."


def test_default_verbosity_is_medium():
    assert estimate_speaking_duration(SAMPLE_TEXT) == estimate_speaking_duration(SAMPLE_TEXT, "Medium")


def test_unrecognized_verbosity_falls_back_to_medium():
    assert estimate_speaking_duration(SAMPLE_TEXT, "Nonexistent") == estimate_speaking_duration(SAMPLE_TEXT, "Medium")


def test_matches_fitted_rate_and_delay_per_tier():
    syllables = count_syllables(SAMPLE_TEXT)
    for tier in ("Low", "Medium", "High"):
        expected = syllables / (SPEECH_RATE_SPM_BY_VERBOSITY[tier] / 60) + SPEECH_DELAY_BY_VERBOSITY[tier]
        assert estimate_speaking_duration(SAMPLE_TEXT, tier) == pytest.approx(expected)


def test_delay_increases_with_verbosity():
    # An empty answer's duration is pure delay.
    assert estimate_speaking_duration("", "Low") < estimate_speaking_duration("", "Medium") < estimate_speaking_duration("", "High")


def test_rate_increases_with_verbosity():
    # For a long enough answer, High's faster rate outweighs its larger fixed delay.
    long_text = SAMPLE_TEXT * 20
    low = estimate_speaking_duration(long_text, "Low")
    medium = estimate_speaking_duration(long_text, "Medium")
    high = estimate_speaking_duration(long_text, "High")
    assert low > medium > high
