"""
MimiTalk generator — reproduced from baselines/mimitalk/mimitalk.py.

Algorithm: Dual-agent interview. A supervisor LLM periodically analyzes the
conversation and provides strategic guidance; a responder LLM uses that guidance
to ask the next question. Both are AI-driven (no rigid topic loop).

Supervisor trigger logic (from original):
  - Always on the first two user messages
  - Any time recent responses are very short (< 10 chars)
  - Every SUPERVISOR_FREQUENCY user messages

Dependencies:
  - openai >= 1.0  (pip install openai)
  - OPENAI_API_KEY environment variable

Configuration: Edit the constants below before use.
  DEFAULT_MODEL          — responder model
  DEFAULT_SUPERVISOR_MODEL — supervisor model (can differ from responder)
  SUPERVISOR_FREQUENCY   — call supervisor every N user messages (default 3)
  SUPERVISOR_MAX_TOKENS  — max tokens for supervisor response
  MAX_TURNS              — hard stop after this many turns
  self.interview_spec         — topics/subtopics to cover
"""

import json
from typing import Optional, Tuple, List
from openai import OpenAI

from . import _end_interview

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_SUPERVISOR_MODEL = "gpt-5.4-mini"

def _temp_kwarg(value, model=None):
    m = model if model is not None else DEFAULT_MODEL
    return {"temperature": value} if m == "gpt-4.1-mini" else {}
SUPERVISOR_FREQUENCY = 3
SUPERVISOR_MAX_TOKENS = 8192
MAX_TURNS = 500  # safety backstop — primary termination is time-based (see sessions.py)

# ── JSON helper ────────────────────────────────────────────────────────────────

def _extract_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(s[start:end])
        except Exception:
            pass
    raise ValueError(f"Could not extract valid JSON from model response: {s[:200]}")


# ── Supervisor trigger (verbatim from baselines/mimitalk/mimitalk.py L166-196) ─

def _should_trigger_supervisor(history: List[dict], frequency: int = 3) -> bool:
    user_messages = [msg for msg in history if msg.get("role") == "user"]
    num_user_messages = len(user_messages)

    # Detect short/repetitive responses
    if len(history) >= 4:
        recent_messages = history[-4:]
        user_responses = [msg["content"] for msg in recent_messages if msg.get("role") == "user"]
        if len(user_responses) >= 2:
            if any(len(resp.strip()) < 10 for resp in user_responses[-2:]):
                return True

    # Always trigger in initial phase
    if num_user_messages <= 1:
        return True

    # Frequency control
    if num_user_messages % frequency != 0:
        return False

    return True


# ── Supervisor call (verbatim prompt from baselines/mimitalk/mimitalk.py L224-250) ─

def _call_supervisor_sync(
    spec: dict,
    history: List[dict],
    client: OpenAI,
    supervisor_model: str,
    max_tokens: int = SUPERVISOR_MAX_TOKENS,
) -> Tuple[str, int]:
    recent_history = history[-10:] if len(history) > 10 else history

    supervisor_prompt = f"""You are an AI interview supervision expert, analyzing interview quality and providing strategic guidance.

**Full Interview Guide** (all topics and subtopics):
{json.dumps(spec, indent=2)}

**Interview Type**: Semi-structured / Flexible (AI-driven progression)

**Analysis Dimensions**:
1. Interview depth and quality
2. Interviewee engagement level
3. Topic coverage completeness across ALL topics
4. Conversation flow and natural transitions
5. Follow-up opportunities

**Your Task**:
Analyze the conversation history and provide strategic guidance:
- Which topics/subtopics have been covered adequately
- Whether to probe deeper or transition to new areas
- Quality of information gathered so far
- Suggested angles, follow-ups, or transitions to pursue
- Coverage gaps that should be addressed

**Note**: The interviewer AI will decide the next question - your role is strategic guidance only.

**Conversation History**:
{json.dumps(recent_history, indent=2)}
"""

    response = client.chat.completions.create(
        model=supervisor_model,
        **_temp_kwarg(0.7, supervisor_model),
        messages=[{"role": "user", "content": supervisor_prompt}],
        max_completion_tokens=max_tokens,
    )
    analysis = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if hasattr(response, "usage") else 0
    return (analysis, tokens_used)


# ── InterviewerAgent ──────────────────────────────────────────────────────────────────

class InterviewerAgent:
    def __init__(self, pre_survey_data, study_id, session_id, temperature=0.7, interview_description="", interview_spec=None):
        self.pre_survey_data = pre_survey_data
        self.study_id = study_id
        self.session_id = session_id
        self.temperature = temperature
        self.interview_spec = interview_spec or []

        self.client = OpenAI()
        self.data_source_text = "\n".join(
            f"{k}: {v}" for k, v in pre_survey_data.items()
        ) if pre_survey_data else ""

        self.accumulated_notes: list = []
        self.closing_message: Optional[str] = None

    def generate_question(self, turn_number, current_turn, full_transcript) -> Optional[str]:
        if turn_number > MAX_TURNS:
            return None

        # The end_interview field is only offered once enough interview time has passed.
        can_end = _end_interview.end_allowed(self)

        # Rebuild history from completed turns
        history = []
        for t in full_transcript:
            history.append({"role": "assistant", "content": t["question_text"]})
            history.append({"role": "user", "content": t["transcript"]})

        # Append current (unanswered) user turn if present
        if current_turn:
            history.append({"role": "user", "content": current_turn})

        full_spec_json = json.dumps(self.interview_spec, indent=2)

        # ── Supervisor step ────────────────────────────────────────────────────
        supervisor_analysis = None
        if _should_trigger_supervisor(history, SUPERVISOR_FREQUENCY):
            supervisor_analysis, _ = _call_supervisor_sync(
                spec=self.interview_spec,
                history=history,
                client=self.client,
                supervisor_model=DEFAULT_SUPERVISOR_MODEL,
                max_tokens=SUPERVISOR_MAX_TOKENS,
            )

        # ── Responder system prompt (verbatim from mimitalk.py L335-391) ──────
        if supervisor_analysis:
            system_prompt = f"""You are a professional AI interviewer conducting an in-depth, conversational interview.

**Supervisor's Strategic Guidance**:
{supervisor_analysis}

**Full Interview Guide** (all topics and subtopics):
{full_spec_json}

**Your Task**:
You have full autonomy to conduct the interview naturally. Based on the conversation history and supervisor guidance:
- Decide what topic/subtopic to explore next based on conversation flow
- Determine whether to probe deeper on current topic or transition to new areas
- Ask exactly ONE question per turn
- Paraphrase subtopics into conversational questions (never use subtopic text verbatim)
- Build on prior answers to maintain natural flow
- Cover topics in the guide over the course of the interview

**Notes Capture** (REQUIRED):
Generate structured notes from the user's LAST response:
- Concise (1-2 sentences max)
- Factual: dates, names, metrics, technologies, achievements
- Third person (e.g., "Worked at X from 2020-2022...")

**Output Format** (strict JSON):
{{
  "question_to_ask": "<your next question>",
  "notes": "<structured notes from user's last response>"
}}

Output ONLY the JSON, no other text."""
        else:
            system_prompt = f"""You are a professional AI interviewer conducting an in-depth, conversational interview.

**Full Interview Guide** (all topics and subtopics):
{full_spec_json}

**Your Task**:
Conduct a natural interview, deciding what to ask based on conversation history:
- Ask exactly ONE question per turn
- Decide what to explore based on conversation flow and guide coverage
- Paraphrase subtopics into conversational questions
- Build natural flow and transitions

**Notes Capture** (REQUIRED):
Generate structured notes from the user's LAST response:
- Concise (1-2 sentences max)
- Factual: dates, names, metrics, technologies, achievements
- Third person (e.g., "Worked at X from 2020-2022...")

**Output Format** (strict JSON):
{{
  "question_to_ask": "<question>",
  "notes": "<notes>"
}}

Output ONLY the JSON, no other text."""

        if can_end:
            system_prompt += _end_interview.JSON_END_INSTRUCTIONS.format(
                message_field="question_to_ask"
            )

        messages = [{"role": "system", "content": system_prompt}] + history

        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            **_temp_kwarg(self.temperature),
            messages=messages,
            response_format={"type": "json_object"},
        )

        result = _extract_json(response.choices[0].message.content)
        notes = result.get("notes", "")
        if notes:
            self.accumulated_notes.append({
                "turn_number": turn_number,
                "notes": notes,
                "supervisor_used": supervisor_analysis is not None,
            })

        if can_end and result.get("end_interview"):
            self.closing_message = result.get("question_to_ask") or ""
            return None

        return result.get("question_to_ask")
