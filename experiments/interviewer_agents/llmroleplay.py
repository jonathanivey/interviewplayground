"""
LLMRolePlay generator — reproduced from baselines/llmroleplay/llmroleplay.py.

Algorithm: Structured nested loop — topic → subtopic. After each participant
answer the model decides whether the response was sufficient ("ask_next") or
should be probed further ("reask"). Unsatisfied subtopics are re-asked up to
MAX_REASKS times before forcing progression to the next subtopic.

Two kinds of LLM calls per subtopic:
  1. Initial ask  — called with latest_user_answer=None to generate the opening
                    question for a subtopic (turn 1 and each new topic start).
  2. Decision call — called with the participant's answer to decide
                    ask_next/reask and generate the next question.

Dependencies:
  - openai >= 1.0  (pip install openai)
  - OPENAI_API_KEY environment variable

Configuration: Edit the constants below before use.
  DEFAULT_MODEL   — interviewer model
  MAX_TURNS       — hard stop (set higher than topic count × avg turns/subtopic)
  MAX_REASKS      — re-ask ceiling per subtopic before forced progression
  self.interview_spec  — loaded from config.py; format:
                    [{"topic": "...", "subtopics": ["...", ...]}]
                    Note: subtopics are AREAS, not literal questions.
                    The model paraphrases them into natural conversational questions.
"""

import json
from typing import Optional, List
from openai import OpenAI

from . import _end_interview

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-5.4-mini"
MAX_TURNS = 500  # safety backstop — primary termination is time-based (see sessions.py)
MAX_REASKS = 3

def _temp_kwarg(value):
    return {"temperature": value} if DEFAULT_MODEL == "gpt-4.1-mini" else {}

# ── System prompt (verbatim from baselines/llmroleplay/llmroleplay.py L248-285) ─

SYSTEM_PROMPT_CV = """You are an AI interviewer designed to collect detailed and structured information about a candidate's professional background, similar to what would appear on a CV or résumé. Be polite and affable in your tone while formal in your approach to reconstruct the person's background entirely.

Your goal is to ask clear, specific, and adaptive questions that help you understand the background of the candidate. Ask simple clear questions so that you don't overwhelm the candidate. Make the conversation flow naturally, building on prior answers. If they have answered something already before, avoid repeating it. You are required to ask at most ONE QUESTION at a time.

If a data source is provided, treat it as partial information about the candidate's CV. Use it to personalize your phrasing and fill gaps, for example:
- Instead of "Where did you study?", ask "I see you completed your MTech at MIT — what was your focus area there?"
- Instead of "Tell me about your last role," ask "You mentioned working at Infosys as a specialist programmer — what kind of projects were you handling?"
- If the subtopic you are asking relates to multiple aspects of the CV (several jobs, skills or time periods), ensure you cover each one, either one at a time or ask a follow-up to other points where it might be relevant. Your goal is to obtain a complete picture of the candidate's background.
- If something is completely covered in the data source, avoid asking about it again. Don't be repetitive.

You must:
- Ask exactly **one** question per turn.
- Keep your tone professional, focused, and curious — like a recruiter collecting detailed information, not a casual chat.
- Rephrase or re-ask when the response lacks specificity (e.g., missing time periods, tasks, tools, or metrics).
- Summarize only when explicitly asked (e.g., "This is what we got so far…").

When a response seems vague, re-ask the same question in a more concrete, guiding way (e.g., "Could you give an example of one project and roughly how much time it took?").

When "next_subtopic_if_any" is null and your decision is "ask_next", there is nothing left to cover. In that case "assistant_message" must be a brief, warm closing statement that thanks the candidate and tells them the interview is finished — not a question.

IMPORTANT: You will receive a "subtopic" (not a fully-formed question). You must paraphrase this subtopic into a SINGLE natural, conversational question that explores that area. DO NOT exactly output the original subtopic.
You must also generate brief, structured notes that capture the key information provided. These notes should be:
- Concise (1-2 sentences max)
- Factual and structured
- Include key details: dates, names, metrics, technologies, achievements
- Written in third person (e.g., "Worked at X from Y to Z...")
- Include previous notes collected

Your output **must** be a valid JSON object following this schema:
{
  "assistant_message": "<the exact next question to ask, must paraphrase from subtopic and only one question>",
  "satisfied": true or false,
  "decision": "ask_next" or "reask",
  "reason": "<brief reason for whether you are re-asking or proceeding>",
  "question_to_ask": "<reworked same question if reask, or next question paraphrased from next subtopic>",
  "notes": "<brief structured notes capturing key info from user's last response and previous notes>"
}

Do not include code fences, commentary, or additional text outside this JSON.
"""

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

        # Progression state
        self.topic_idx = 0
        self.question_idx = 0
        self.reask_counter = 0
        # topic_start_turn: index into full_transcript where the current topic began.
        # Slicing full_transcript from here gives current-topic history only,
        # matching the original which resets conversation history per topic.
        self.topic_start_turn = 1
        # True on the very first turn and whenever we enter a new topic.
        # When True, generate_question makes an "initial ask" (latest_user_answer=None)
        # rather than a "decision call".
        self._is_first_ask = True

        self.accumulated_notes: list = []
        self.closing_message: Optional[str] = None

    # ── Internal LLM call ─────────────────────────────────────────────────────

    def _call_model(
        self,
        topic_name: str,
        current_subtopic: str,
        next_subtopic: Optional[str],
        latest_user_answer: Optional[str],
        topic_history: List[dict],
        can_end: bool = False,
    ) -> dict:
        notes_context = ""
        relevant = [
            n for n in self.accumulated_notes
            if n.get("subtopic") == current_subtopic
        ]
        if relevant:
            notes_context = "\n\nPreviously captured information on same subtopic:\n"
            for n in relevant:
                notes_context += f"- {n['subtopic']}: {n['notes']}\n"

        user_payload = json.dumps(
            {
                "context": {
                    "topic_name": topic_name,
                    "current_subtopic": current_subtopic,
                    "next_subtopic_if_any": next_subtopic,
                    "conversation_history": topic_history,
                    "latest_user_answer": latest_user_answer,
                    "data_source_text": self.data_source_text,
                    "previously_captured_notes": notes_context,
                    "notes": [
                        "The 'current_subtopic' and 'next_subtopic_if_any' are NOT questions—they are topic areas you need to explore.",
                        "Paraphrase the subtopic into a natural, conversational question.",
                        "Ask exactly one question next. Do not repeat any prior questions.",
                        "If the subtopic is answered sufficiently, generate brief notes and proceed to next_subtopic_if_any (or close if none).",
                        "If the answer is too brief and has insufficient detail, re-ask about the same subtopic more concretely.",
                        "You MUST include 'notes' field with condensed information from user's response.",
                        "Notes should be concise (1-2 sentences), factual, and capture key details like dates, names, metrics, technologies.",
                        "Provide a reason for your decision in the 'reason' field.",
                        "Keep wording contextual, referencing prior details briefly.",
                        "Use data_source_background to personalize or adapt the question as relevant.",
                        "Focus on concrete, CV-style details: job titles, durations, responsibilities, tools, and outcomes.",
                        "Avoid hypothetical or motivational questions — keep it factual and descriptive.",
                        "Encourage short structured answers where relevant.",
                        "If the user is adversarial or defensive, stay professional and friendly and just move to the next subtopic",
                    ],
                },
                "required_output_schema": {
                    "assistant_message": "string",
                    "satisfied": "boolean",
                    "decision": "ask_next | reask",
                    "reason": "string",
                    "question_to_ask": "string",
                    "notes": "string",
                    **(_end_interview.JSON_SCHEMA_FIELD if can_end else {}),
                },
            },
            ensure_ascii=False,
            indent=2,
        )

        system_prompt = SYSTEM_PROMPT_CV
        if can_end:
            system_prompt += _end_interview.JSON_END_INSTRUCTIONS.format(
                message_field="assistant_message"
            )

        resp = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            **_temp_kwarg(self.temperature),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            response_format={"type": "json_object"},
        )
        return _extract_json(resp.choices[0].message.content)

    # ── Progression helpers ───────────────────────────────────────────────────

    def _advance(self, turn_number: int):
        """Move to the next subtopic, or next topic, or signal completion."""
        self.question_idx += 1
        self.reask_counter = 0

        current_subtopics = self.interview_spec[self.topic_idx].get("subtopics", [])
        if self.question_idx >= len(current_subtopics):
            # End of current topic — move to next
            self.topic_idx += 1
            self.question_idx = 0
            self.topic_start_turn = turn_number
            self._is_first_ask = True

    def _is_done(self) -> bool:
        return self.topic_idx >= len(self.interview_spec)

    # ── Public interface ──────────────────────────────────────────────────────

    def generate_question(self, turn_number, current_turn, full_transcript) -> Optional[str]:
        if self._is_done() or turn_number > MAX_TURNS:
            return None

        topic = self.interview_spec[self.topic_idx]
        topic_name = topic.get("topic", f"Topic {self.topic_idx + 1}")
        subtopics = topic.get("subtopics", [])

        if not subtopics or self.question_idx >= len(subtopics):
            return None

        current_subtopic = subtopics[self.question_idx]

        # Determine next_subtopic: within the same topic first; if last in topic,
        # peek at first subtopic of the next topic so the model can transition.
        if (self.question_idx + 1) < len(subtopics):
            next_subtopic = subtopics[self.question_idx + 1]
        elif (self.topic_idx + 1) < len(self.interview_spec):
            next_topic = self.interview_spec[self.topic_idx + 1]
            next_subs = next_topic.get("subtopics", [])
            next_subtopic = (
                f"[Next topic: {next_topic.get('topic', '')}] {next_subs[0]}"
                if next_subs else None
            )
        else:
            next_subtopic = None

        # Build history for the current topic only (mirrors the original's per-topic reset)
        topic_history = []
        for t in full_transcript[self.topic_start_turn - 1:]:
            topic_history.append({"role": "assistant", "content": t["question_text"]})
            topic_history.append({"role": "user", "content": t["transcript"]})

        # ── Initial ask ───────────────────────────────────────────────────────
        if self._is_first_ask:
            self._is_first_ask = False
            result = self._call_model(
                topic_name=topic_name,
                current_subtopic=current_subtopic,
                next_subtopic=None,
                latest_user_answer=None,
                topic_history=topic_history,
            )
            return result.get("question_to_ask") or current_subtopic

        # ── Decision call ─────────────────────────────────────────────────────
        # The end_interview field is only offered once enough interview time has passed.
        can_end = _end_interview.end_allowed(self)

        result = self._call_model(
            topic_name=topic_name,
            current_subtopic=current_subtopic,
            next_subtopic=next_subtopic,
            latest_user_answer=current_turn,
            topic_history=topic_history,
            can_end=can_end,
        )

        notes = result.get("notes", "")
        if notes:
            self.accumulated_notes.append({
                "topic": topic_name,
                "subtopic": current_subtopic,
                "notes": notes,
            })

        decision = result.get("decision", "reask")
        next_question = result.get("assistant_message") or result.get("question_to_ask", "")

        if can_end and result.get("end_interview"):
            self.closing_message = next_question
            return None

        if decision == "ask_next":
            self._advance(turn_number)
            if self._is_done():
                # Spec exhausted. next_subtopic was null on this call, so the model
                # was asked to make next_question a closing statement.
                self.closing_message = next_question
                return None
        else:
            self.reask_counter += 1
            if self.reask_counter >= MAX_REASKS:
                self._advance(turn_number)
                if self._is_done():
                    # next_question here is a re-asked question, not a closing
                    # statement, so leave closing_message unset.
                    return None

        return next_question if next_question else None
