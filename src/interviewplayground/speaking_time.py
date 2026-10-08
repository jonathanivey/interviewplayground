"""Estimate how long a piece of text would take to speak aloud.

The estimate counts syllables (via textstat.syllable_count, which uses the
Pyphen hyphenation dictionaries), divides by a speech rate, and adds a fixed
per-response delay for the time it takes to start answering. Both the rate
and the delay depend on the participant's Verbosity trait ("Low"/"Medium"/
"High", see prompts.TRAIT_DESCRIPTIONS): participants who give longer answers
also articulate faster, but take longer to start answering.
"""

import textstat

SPEECH_RATE_SPM_BY_VERBOSITY = {
    "Low": 167.7,
    "Medium": 189.5,
    "High": 216.1,
}
SPEECH_DELAY_BY_VERBOSITY = {
    "Low": 11.3,
    "Medium": 16.0,
    "High": 26.6,
}


def count_syllables(text: str) -> int:
    """Total syllables in the input text."""
    return textstat.syllable_count(text)


def estimate_speaking_duration(text: str, verbosity: str = "Medium") -> float:
    """Estimate how long the given text would take to speak aloud, in seconds.

    Counts syllables (via textstat.syllable_count), divides by the fitted
    speech rate for the given Verbosity tier, and adds that tier's fitted
    per-response delay. Falls back to the "Medium" tier for an unrecognized
    verbosity value.
    """
    rate_spm = SPEECH_RATE_SPM_BY_VERBOSITY.get(verbosity, SPEECH_RATE_SPM_BY_VERBOSITY["Medium"])
    delay = SPEECH_DELAY_BY_VERBOSITY.get(verbosity, SPEECH_DELAY_BY_VERBOSITY["Medium"])
    return count_syllables(text) / (rate_spm / 60) + delay