"""
Shared end-interview capability for the five study interviewer agents.

Each of interviewgpt, mimitalk, llmroleplay, sparkme and storysage gets a tool the
model can call when it judges the interview complete. Two prompt formats are provided
because the agents split into two families:

  - JSON output (interviewgpt, mimitalk, llmroleplay): an "end_interview" boolean
    added to the required output schema.
  - Pseudo-tool output (sparkme, storysage): an <end_interview> tool called instead
    of respond_to_user.

The tool is only offered once the session has accumulated MIN_SECONDS_BEFORE_END of
effective interview time. Below that floor these fragments are left out of the prompt
entirely, so the model has no way to end early. Effective time is set on the agent
instance as `elapsed_seconds` by researcher/generate_question.py before each turn.

An agent that ends sets `self.closing_message` to the model's closing statement and
returns None from generate_question. The backend reads it back via
researcher.generate_question.get_closing_message().
"""

import re

# 10 minutes. The full budget is MAX_INTERVIEW_SECONDS = 1800 (backend/routers/sessions.py).
MIN_SECONDS_BEFORE_END = 600


def end_allowed(agent) -> bool:
    """True once the session has enough effective time on the clock to offer the
    end-interview tool to the model."""
    return getattr(agent, "elapsed_seconds", 0.0) >= MIN_SECONDS_BEFORE_END


# ── JSON-output agents (interviewgpt, mimitalk, llmroleplay) ──────────────────

# Added to the "required_output_schema" the agents send in their user payload.
JSON_SCHEMA_FIELD = {"end_interview": "boolean"}

# Appended to the system prompt. {message_field} is the agent's own message key.
JSON_END_INSTRUCTIONS = """

# Ending the interview

Your output schema has one additional field:
  "end_interview": true or false

Set "end_interview" to true only when the interview is definitely complete, every topic in the interview guide has been
covered, and the participant has nothing further to add. When you set it to true,
"{message_field}" must be a brief, warm closing statement — thank the participant and
tell them the interview is finished — and must NOT be a question. Otherwise set
"end_interview" to false and ask your next question as usual.
"""


# ── Pseudo-tool agents ────────────────────────────────────────────────────────

# XML tool block, matching format_tool_as_xml_v2 output (sparkme).
XML_TOOL_DESCRIPTION = """
<end_interview>
  <description>
    End the interview. Call this INSTEAD of respond_to_user, and only when the interview is definitely complete, every topic in the interview guide has been
covered, and the participant has nothing further to add. Once you call this the interview is over and the user cannot reply.
  </description>
  <arguments>
    <closing_message>
      <type>str</type>
      <description>
        A brief, warm closing statement thanking the user and telling them the interview is finished. Not a question.
      </description>
    </closing_message>
  </arguments>
</end_interview>"""

# Plain-text tool description (storysage).
TEXT_TOOL_DESCRIPTION = (
    "\n\nend_interview: Use this tool to end the interview.\n"
    "  Call this INSTEAD of respond_to_user, and only when the interview is definitely complete, every topic in the interview guide has been covered, and the participant has nothing further to add.\n"
    "  Once you call this the interview is over and the user cannot reply.\n"
    "  Arguments:\n"
    "    - closing_message (str): A brief, warm closing statement thanking the user\n"
    "      and telling them the interview is finished. Not a question."
)

# Appended as its own section after the agent's existing output-format block.
XML_OUTPUT_FORMAT = """
<output_format_end_interview>

If the interview is complete, call end_interview INSTEAD of respond_to_user, in the same format:
<tool_calls>
  <end_interview>
    <closing_message>value</closing_message>
  </end_interview>
</tool_calls>

</output_format_end_interview>"""

# Added to the prompt when the agent's own agenda tracking says nothing is left.
# Without it the agents' "always find a follow-up" instructions win and they never
# call the tool, however thoroughly the guide has been covered.
AGENDA_EXHAUSTED_NOTE = (
    "Every item on the interview agenda has been covered — there is nothing left "
    "to explore."
)

END_TAG_RE = re.compile(
    r"<end_interview>\s*<closing_message>(.*?)</closing_message>\s*</end_interview>",
    re.DOTALL,
)
