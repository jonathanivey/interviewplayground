"""
InterviewGPT generator — reproduced from baselines/interviewgpt/interviewgpt.py.

Algorithm: Free-form, AI-driven interview. One LLM call per turn. The model sees
the full conversation history plus accumulated notes and freely decides the next
question based on the spec topics.

Dependencies:
  - openai >= 1.0  (pip install openai)
  - OPENAI_API_KEY environment variable

Configuration: Edit the constants below before use.
  DEFAULT_MODEL   — interviewer model
  MAX_TURNS       — hard stop after this many turns
  INTERVIEW_SPEC  — loaded from config.py; format:
                    [{"topic": "...", "subtopics": ["...", ...]}, ...]
"""

import json
import re
from typing import Optional
from openai import OpenAI

from . import _end_interview

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-5.4-mini"
MAX_TURNS = 500  # safety backstop — primary termination is time-based (see sessions.py)

def _temp_kwarg(value):
    return {"temperature": value} if DEFAULT_MODEL == "gpt-4.1-mini" else {}

# ── System prompt (verbatim from baselines/interviewgpt/interviewgpt.py L219-270) ─

SYSTEM_PROMPT_CV = """# Your role as an AI interviewer

You are a survey interviewer named 'InterviewGPT', an AI interviewer. You are a highly skilled Interviewer AI, specialized in conducting qualitative research with the utmost professionalism.
Your programming includes a deep understanding of ethical interviewing guidelines, ensuring your questions are non-biased, non-partisan, and designed to elicit rich, insightful responses.
You navigate conversations with ease, adapting to the flow while maintaining the research's integrity.
You are a professional interviewer that is well trained in interviewing people and takes into consideration the guidelines from recent research to interview people and retrieve information.
Try to ask question that are not biased. The following is really important: If they answer in very short sentences ask follow up questions to gain a better understanding what they mean or ask them to elaborate their view further.
Try to avoid direct questions on intimate topics and assure them that their data is handled with care and privacy is respected.

# Guidelines for asking questions

It is Important to ask one question at a time. Make sure that your questions do not guide or predetermine the respondents' answers in any way.
Do not provide respondents with associations, suggestions, or ideas for how they could answer the question.
If the respondents do not know how to answer a question, move to the next question. Do not judge the respondents' answers.
Do not take a position on whether their answers are right or wrong. Yet, do ask neutral follow-up questions for clarification in case of surprising, unreasonable or nonsensical questions.
You should take a casual, conversational approach that is pleasant, neutral, and professional. It should neither be overly cold nor overly familiar.
From time to time, restate concisely in one or two sentences what was just said, using mainly the respondent's own words.
Then you should ask whether you properly understood the respondents' answers. Importantly, ask follow-up questions when a respondent gives a surprising, unexpected or unclear answer.
Prompting respondents to elaborate can be done in many ways. You could ask: "Why is that?", "Could you expand on that?", "Anything else?", "Can you give me an example that illustrates what you just said?".
Make it seem like a natural conversation. When it makes sense, try to connect the questions to the previous answer.
Try to elicit as much information as possible about the answers from the users; especially if they only provide short answers.
You should begin the interview based on the first question in the questionnaire below. You should finish the interview after you have asked all the questions from the questionnaire.
It is very important to ask only one question at a time, do not overload the interviewee with multiple questions.
Ask the questions precisely and short like in a conversation, with instructions or notes for the interviewer where necessary.
Consider incorporating sections or themes if the questions cover distinct aspects of the topic.

# Interview Outlines

{outlines}

# Instructions

You are conducting an interview to gather detailed information about {interview_description} from an interviewee. Your goal is to ask one precise question at a time, based on the subtopics provided in the interview outlines.
You have to strictly paraphrase the subtopics from the outlines where you should not copy the subtopic directly into your question.
For example, if the subtopic is "Experience with AI tools", you could ask "Have you used any AI tools in your daily work?" instead of "Can you describe your experience with AI tools?".

Avoid asking multiple questions at once; focus on one aspect per question.
You must generate brief, structured notes that capture the key information provided. These notes should be:
- Concise (1-2 sentences max)
- Factual and structured
- Include key details: dates, names, metrics, technologies, achievements
- Written in third person (e.g., "Worked at X from Y to Z...")
- Include previous notes collected

Your output **must** be a valid JSON object following this schema:
{{
    "assistant_message": "<The question to be asked>",
    "notes": "<brief structured notes capturing key info from user's last response>"
}}

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
    def __init__(self, pre_survey_data, study_id, session_id, temperature=0, interview_description="", interview_spec=None):
        self.pre_survey_data = pre_survey_data
        self.study_id = study_id
        self.session_id = session_id
        self.temperature = temperature

        self.client = OpenAI()
        self.system_prompt = SYSTEM_PROMPT_CV.format(
            outlines=json.dumps(interview_spec or [], indent=2),
            interview_description=interview_description,
        )
        # Serialize participant background for use as data_source_text
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

        # Build conversation history from completed turns
        history = []
        for t in full_transcript:
            history.append({"role": "assistant", "content": t["question_text"]})
            history.append({"role": "user", "content": t["transcript"]})

        latest_user_answer = current_turn if current_turn else None

        # Build previously captured notes context
        notes_context = ""
        if self.accumulated_notes:
            notes_context = "\n\nPreviously captured information:\n"
            for note in self.accumulated_notes:
                notes_context += f"- {note}\n"

        user_payload = json.dumps(
            {
                "context": {
                    "conversation_history": history,
                    "latest_user_answer": latest_user_answer,
                    "data_source_text": self.data_source_text,
                    "previously_captured_notes": notes_context,
                    "notes": [
                        "Ask exactly one question next. Do not repeat any prior questions.",
                        "You MUST include 'notes' field with condensed information from user's response.",
                        "Notes should be concise (1-2 sentences), factual, and capture key details like dates, names, metrics, technologies.",
                        "Keep wording contextual, referencing prior details briefly.",
                        "Avoid hypothetical or motivational questions — keep it factual and descriptive.",
                        "Encourage short structured answers where relevant.",
                    ],
                },
                "required_output_schema": {
                    "assistant_message": "string",
                    "notes": "string",
                    **(_end_interview.JSON_SCHEMA_FIELD if can_end else {}),
                },
            },
            ensure_ascii=False,
            indent=2,
        )

        system_prompt = self.system_prompt
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

        result = _extract_json(resp.choices[0].message.content)
        notes = result.get("notes", "")
        if notes:
            self.accumulated_notes.append(notes)

        if can_end and result.get("end_interview"):
            self.closing_message = result.get("assistant_message") or ""
            return None

        return result.get("assistant_message")
