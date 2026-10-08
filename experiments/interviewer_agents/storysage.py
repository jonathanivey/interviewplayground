"""
StorySage generator — normal prompt mode with session agenda infrastructure.

Algorithm: AI-driven interview using engagement scoring and follow-up strategy.
The model evaluates engagement, follows up on high-engagement responses, and
uses a session agenda (question bank) for topic progression.

Session agenda mirrors the in-memory portion of:
  baselines/storysage/content/session_agenda/session_agenda.py
  baselines/storysage/content/session_agenda/interview_question.py

The question bank is built from the study's interview_spec (topic/subtopics),
so StorySage covers the same topics as the other structured interviewers.

Answered-question tracking: the interviewer uses an add_note tool call to
annotate questions after each user answer (parsed in generate_question), which
marks agenda questions as covered so they are hidden from later prompts.

Dependencies:
  - openai >= 1.0  (pip install openai)
  - OPENAI_API_KEY environment variable

Configuration: Edit the constants below before use.
  DEFAULT_MODEL — model to use
  MAX_TURNS     — hard stop after this many turns
"""

import re
from typing import Optional
from openai import OpenAI

from . import _end_interview

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-5.4-mini"
MAX_TURNS = 500  # safety backstop — primary termination is time-based (see sessions.py)

def _temp_kwarg(value):
    return {"temperature": value} if DEFAULT_MODEL == "gpt-4.1-mini" else {}

# ── Session Agenda Infrastructure ──────────────────────────────────────────────
# Mirrors content/session_agenda/interview_question.py and the in-memory
# portions of content/session_agenda/session_agenda.py.

class InterviewQuestion:
    def __init__(self, topic: str, question_id: str, question: str):
        self.topic = topic
        self.question_id = question_id
        self.question = question
        self.notes: list[str] = []
        self.sub_questions: list['InterviewQuestion'] = []

    def serialize(self) -> dict:
        return {
            "topic": self.topic,
            "question_id": self.question_id,
            "question": self.question,
            "notes": self.notes,
            "sub_questions": [sq.serialize() for sq in self.sub_questions],
        }


class SessionAgenda:
    """Lightweight in-memory session agenda (no file I/O).

    Mirrors the core interface of SessionAgenda from the original repo,
    omitting persistence (load_from_file, save, get_last_session_agenda)
    and the historical-summaries helper that require a filesystem layout
    not available in this standalone interface.

    The question bank is built from the study's interview_spec — a list of
    {"topic": str, "subtopics": [str, ...]} dicts — so the topics covered
    match the study configuration rather than a hardcoded script.
    """

    def __init__(self, pre_survey_data: dict, interview_spec: list):
        self.user_portrait: dict = pre_survey_data or {}
        self.last_meeting_summary: str = (
            "This is the first session with the user. "
            "We will start by getting to know them and "
            "understanding their background."
        )
        self.topics: dict[str, list[InterviewQuestion]] = {}
        self.additional_notes: list[str] = []

        question_id = 1
        for entry in interview_spec or []:
            topic = entry["topic"]
            for question in entry.get("subtopics", []):
                self.add_interview_question(topic, question, question_id=str(question_id))
                question_id += 1

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def add_interview_question(self, topic: str, question: str, question_id: str):
        if not question_id:
            raise ValueError("question_id is required")
        if '.' not in question_id:
            if topic not in self.topics:
                self.topics[topic] = []
            self.topics[topic].append(InterviewQuestion(topic, question_id, question))
        else:
            parent_id = question_id.rsplit('.', 1)[0]
            parent = self.get_question(parent_id)
            if not parent:
                raise ValueError(f"Parent question with id {parent_id} not found")
            parent.sub_questions.append(InterviewQuestion(topic, question_id, question))

    def add_note(self, question_id: str = "", note: str = ""):
        if not note:
            return
        if question_id:
            question = self.get_question(question_id)
            if question:
                question.notes.append(note)
        else:
            self.additional_notes.append(note)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get_question(self, question_id: str) -> Optional[InterviewQuestion]:
        topic = None
        for t, questions in self.topics.items():
            for q in questions:
                if q.question_id == question_id.split('.')[0]:
                    topic = t
                    break
            if topic:
                break
        if not topic:
            return None
        if '.' not in question_id:
            return next((q for q in self.topics[topic] if q.question_id == question_id), None)
        parts = question_id.split('.')
        current = next((q for q in self.topics[topic] if q.question_id == parts[0]), None)
        for part in parts[1:]:
            if not current:
                return None
            current = next(
                (q for q in current.sub_questions if q.question_id.endswith(part)), None
            )
        return current

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_qa(self, qa: InterviewQuestion, hide_answered: str = "") -> list[str]:
        if hide_answered not in ["", "a", "qa"]:
            raise ValueError('hide_answered must be "", "a", or "qa"')
        lines = []
        if not qa.question:
            pass
        elif qa.notes:
            if hide_answered == "qa":
                lines.append(f"\n[ID] {qa.question_id}: (Answered)")
            else:
                lines.append(f"\n[ID] {qa.question_id}: {qa.question}")
                if hide_answered != "a":
                    for note in qa.notes:
                        lines.append(f"[note] {note}")
        else:
            lines.append(f"\n[ID] {qa.question_id}: {qa.question}")
        for sub_qa in qa.sub_questions:
            lines.extend(self.format_qa(sub_qa, hide_answered=hide_answered))
        return lines

    def get_questions_and_notes_str(self, hide_answered: str = "") -> str:
        if not self.topics:
            return ""
        output = []
        for topic, questions in self.topics.items():
            output.append(f"\nTopic: {topic}")
            for qa in questions:
                output.extend(self.format_qa(qa, hide_answered=hide_answered))
        return "\n".join(output)

    def all_answered(self) -> bool:
        """True when every question in the bank has at least one note."""
        return bool(self.topics) and all(
            q.notes for questions in self.topics.values() for q in questions
        )

    def get_user_portrait_str(self) -> str:
        if not self.user_portrait:
            return ""
        return "\n".join(
            f"{k.replace('_', ' ').title()}: {v}" for k, v in self.user_portrait.items()
        )

    def get_last_meeting_summary_str(self) -> str:
        return self.last_meeting_summary or ""


# ── Prompts (from baselines/storysage/agents/interviewer/prompts.py, normal mode) ──

_CONTEXT = """
<interviewer_persona>
You are a friendly and curious interviewer..
You ask clear, structured questions, but in a conversational and relaxed way — like chatting with a colleague over coffee.
You guide the conversation toward professional background, technical expertise, project involvement, and problem-solving approaches, but you don't sound rigid or scripted.
Your goal is to gather reliable, detailed insights while making the user feel comfortable sharing their experiences and perspectives.
</interviewer_persona>

<context>
Right now, you are conducting an interview with the user about {interview_description}.
</context>
"""

_USER_PORTRAIT = """
Here is some general information that you know about the user:
<user_portrait>
{user_portrait}
</user_portrait>
"""

_LAST_MEETING_SUMMARY = """
Here is a summary of the last interview session with the user:
<last_meeting_summary>
{last_meeting_summary}
</last_meeting_summary>
"""

_CHAT_HISTORY = """
Chat History:
Use the chat history to understand the interview's context and dynamics.
<chat_history>
{chat_history}
</chat_history>


Current Conversation:
Focus on crafting a response to the user's latest message.
Don't repeat phrases and questions same as your recent responses.
Switch to very different topics if the user's explicitly expresses skip the current question.
<current_events>
{current_events}
</current_events>

"""

_QUESTIONS_AND_NOTES = """
Here is a tentative set of topics and questions that you can ask during the interview:
<questions_and_notes>
{questions_and_notes}
</questions_and_notes>
"""

_TOOL_DESCRIPTIONS = """
To be interact with the user, and a memory bank (containing the memories that the user has shared with you in the past), you can use the following tools:
<tool_descriptions>
{tool_descriptions}
</tool_descriptions>
"""

_CONVERSATION_STARTER = """
# Starting the Conversation

Since this is the first round of the interview. Let's begin the interview by greeting the user and explaining your purpose is to learn about the user's work, skills, and experiences with AI in the workforce.

- Reminder: Focus on starting the conversation without proposing follow-up questions. Ensure you cover the above points.
"""

_INSTRUCTIONS = """
Here are a set of instructions that guide you on how to navigate the interview session and take your actions:
<instructions>

# Thinking
- Before taking any actions, analyze the conversation like a friend would:

1. Summarize Current Response
* First identify the last question asked
* Focus on the concrete details they shared:
  -- "They told me about [specific event/place/person]"
  -- "They mentioned [specific detail] that I can ask more about"
* Look for hooks that could lead to more stories:
  -- "They mentioned their friend [name], could ask about that"
  -- "They brought up [place], might have more memories there"
* Notice their enthusiasm about specific parts of the story

2. Score Engagement (1-5)
* High Engagement (4-5) indicators:
  -- Lots of specific details and descriptions
  -- Mentioning other related memories
  -- Enthusiasm about particular moments or details
* Moderate Engagement (3) indicators:
  -- Basic facts but fewer details
  -- Staying on topic but not expanding
* Low Engagement (1-2) indicators:
  -- Very brief or vague responses
  -- Changing the subject
  -- Showing discomfort
  -- Not sharing personal details
  -- Intention to skip the question

3. Review and Reflect on the Conversation
* Check what specific experiences we've already discussed
* Make sure you don't repeat the same questions or phrases
* Look for types of memories they enjoy sharing
* Notice which topics led to good stories by analyzing the people, events, places, and details the user mentioned in the current conversation

# Taking Actions

## 1. Pick Next Question to Ask

### 1.0 Topic Switching
- Switch to a very different topic if the user explicitly requests to skip the current question.

### 1.1 For high engagement stories (4-5):
#### Important Rule: Stay Within Current Context
- When user is highly engaged, only ask about topics, people, or details explicitly mentioned in their last response
- Do NOT revert to a previous topic or introduce new topics while they're engaged with the current story
- Examples:
  * User (enthusiastically): "I went to the beach with Sarah..."
    -- Good: "What did you and Sarah do first when you got there?"
    -- Bad: "Have you been to any other beaches lately?"

#### Follow-up Question Strategy
a. Natural Conversation Flow (Highest Priority)
  - Focus on concrete, easy-to-answer questions about the specific experience
  - Avoid questions that require deep reflection or analysis, such as:
    * "What did you learn from this?"
    * "How did this shape your values?"
    * "What would you do differently?"
    * "How do you think this will affect your future?"
  - Instead, ask for more details about the memory itself:
    * "What was the weather like that day?"
    * "Who else was there with you?"
    * "What did the place look like?"
    * "What happened right after that?"
    * "What did [person they mentioned] say next?"
  - Think of it like helping them paint a picture of the scene
  - Let them naturally share their feelings and reflections if they want to
  - Keep the conversation light and fun, like chatting with a friend

b. Question Bank (Only when current story is fully explored)
  - Use when:
    * You've gotten all the interesting details about the current story
    * User shows low engagement
    * Need to switch topics
  - Choose questions that:
    * Ask about specific experiences or memories
    * Are easy to answer with concrete details
    * Feel natural to the conversation

### 1.2 For moderate engagement (3):
  -- Try more specific questions about details they've mentioned
  -- Can introduce related but different topics if current one feels exhausted

### 1.3 For low engagement (1-2):
  -- Feel free to switch topics completely
  -- Try different types of memories or experiences
  -- Focus on lighter, easier subjects

## 2. Formulate Response
  * First, react to user's previous response with emotional intelligence:
    -- Show genuine empathy for personal experiences
    -- Acknowledge and validate their emotions
    -- Use supportive phrases appropriately:
      * "That must have been [challenging/exciting/difficult]..."
      * "I can understand why you felt that way..."
      * "Thank you for sharing such a personal experience..."
    -- Give them space to process emotional moments
  * Then proceed with:
    * Keep your tone casual and friendly
    * Show interest in the specific details they've shared
    * Connect to concrete details they mentioned earlier when relevant
  * Avoid response patterns:
    * Tailor each response uniquely to what was just shared
    * Match your tone to the emotional content of their message
    * DON'T repeat phrases same as your recent responses:

MOST IMPORTANT:
* 🚨 DON'T repeat phrases same as your recent responses in the following list! 🚨
* This is crucial to maintain diversity in responses and avoid redundancy.
* Pay special attention to avoid phrases like "that sounds like" which have been overused.
<recent_interviewer_messages>
{recent_interviewer_messages}
</recent_interviewer_messages>

{conversation_starter}

## Tools
- Your response should include the tool calls you want to make.
- Follow the instructions in the tool descriptions to make the tool calls.
</instructions>
"""

_OUTPUT_FORMAT = """
<output_format>

Your output should include the tools you need to call according to the following format.
- Wrap the tool calls in <tool_calls> tags as shown below
- No other text should be included in the output like thinking, reasoning, query, response, etc.
- Call add_note (optional) before respond_to_user whenever a question from the bank has been sufficiently answered
<tool_calls>
  <add_note>
    <question_id>value</question_id>
    <note>value</note>
  </add_note>
  <respond_to_user>
      <response>value</response>
  </respond_to_user>
</tool_calls>

</output_format>
"""

# respond_to_user is always called; add_note is optional per turn
_TOOL_DESCRIPTION_TEXT = (
    "respond_to_user: Use this tool to send your response/question to the user.\n"
    "  Arguments:\n"
    "    - response (str): The question or message to deliver to the user.\n\n"
    "add_note: Use this tool to record that a question has been answered.\n"
    "  Call this BEFORE respond_to_user when the user's answer has sufficiently\n"
    "  addressed a question from the question bank.\n"
    "  Arguments:\n"
    "    - question_id (str): The [ID] of the answered question (e.g. \"1\", \"3\").\n"
    "    - note (str): A brief summary of the user's answer (1-2 sentences)."
)


# ── InterviewerAgent ──────────────────────────────────────────────────────────

class InterviewerAgent:
    def __init__(self, pre_survey_data, study_id, session_id, temperature=0, interview_description="", interview_spec=None):
        self.pre_survey_data = pre_survey_data
        self.study_id = study_id
        self.session_id = session_id
        self.temperature = temperature
        self.interview_description = interview_description
        self.client = OpenAI()
        self.session_agenda = SessionAgenda(pre_survey_data, interview_spec)
        self.closing_message: Optional[str] = None

    def generate_question(self, turn_number, current_turn, full_transcript) -> Optional[str]:
        if turn_number > MAX_TURNS:
            return None

        # The end_interview tool is only offered once enough interview time has passed.
        can_end = _end_interview.end_allowed(self)

        # Build event stream and extract interviewer messages
        all_events = []
        all_interviewer_messages = []
        for t in full_transcript:
            all_events.append(f"Interviewer: {t['question_text']}")
            all_interviewer_messages.append(t['question_text'])
            if t['transcript']:
                all_events.append(f"User: {t['transcript']}")
        if current_turn:
            all_events.append(f"User: {current_turn}")

        # Mirrors interviewer._get_prompt(): cap at 30, last 2 are current_events
        recent_events = all_events[-30:]
        current_events = recent_events[-2:] if len(recent_events) >= 2 else recent_events
        chat_history_str = "\n".join(recent_events) if recent_events else "(No prior conversation)"
        current_events_str = "\n".join(current_events)

        # Last 5 interviewer messages, truncated at 120 chars (mirrors original)
        recent_msgs = all_interviewer_messages[-5:]
        recent_interviewer_messages_str = "\n".join(
            msg[:120] + "..." if len(msg) > 150 else msg
            for msg in recent_msgs
        )

        conversation_starter = _CONVERSATION_STARTER if not all_interviewer_messages else ""

        # Pull values from session agenda (mirrors interviewer._get_prompt())
        user_portrait = self.session_agenda.get_user_portrait_str()
        last_meeting_summary = self.session_agenda.get_last_meeting_summary_str()
        questions_and_notes = self.session_agenda.get_questions_and_notes_str(hide_answered="qa")

        # Fill INSTRUCTIONS inner placeholders before assembling
        instructions_filled = _INSTRUCTIONS.format(
            recent_interviewer_messages=recent_interviewer_messages_str,
            conversation_starter=conversation_starter,
        )

        # Assemble prompt matching INTERVIEW_PROMPT order:
        # CONTEXT (system) → USER_PORTRAIT → LAST_MEETING_SUMMARY → CHAT_HISTORY →
        # QUESTIONS_AND_NOTES → TOOL_DESCRIPTIONS → INSTRUCTIONS → OUTPUT_FORMAT
        tool_descriptions = _TOOL_DESCRIPTION_TEXT
        if can_end:
            tool_descriptions += _end_interview.TEXT_TOOL_DESCRIPTION

        sections = [
            _USER_PORTRAIT.format(user_portrait=user_portrait).strip(),
            _LAST_MEETING_SUMMARY.format(last_meeting_summary=last_meeting_summary).strip(),
            _CHAT_HISTORY.format(
                chat_history=chat_history_str,
                current_events=current_events_str,
            ).strip(),
            _QUESTIONS_AND_NOTES.format(questions_and_notes=questions_and_notes).strip(),
            _TOOL_DESCRIPTIONS.format(tool_descriptions=tool_descriptions).strip(),
            instructions_filled.strip(),
        ]
        if can_end and self.session_agenda.all_answered():
            sections.append(_end_interview.AGENDA_EXHAUSTED_NOTE)
        sections.append(_OUTPUT_FORMAT.strip())
        if can_end:
            sections.append(_end_interview.XML_OUTPUT_FORMAT.strip())
        user_message = "\n\n".join(sections)

        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            **_temp_kwarg(self.temperature),
            messages=[
                {"role": "system", "content": _CONTEXT.format(interview_description=self.interview_description).strip()},
                {"role": "user", "content": user_message},
            ],
        )
        content = response.choices[0].message.content

        # Parse and apply any add_note calls before extracting the response
        for note_match in re.finditer(
            r"<add_note>\s*<question_id>(.*?)</question_id>\s*<note>(.*?)</note>\s*</add_note>",
            content, re.DOTALL
        ):
            self.session_agenda.add_note(
                question_id=note_match.group(1).strip(),
                note=note_match.group(2).strip(),
            )

        # end_interview takes precedence over respond_to_user. Parsed unconditionally:
        # the time floor is enforced by leaving the tool out of the prompt, and a call
        # we refused to parse would fall through to the empty-content fallback below
        # and end the interview anyway — without a closing statement.
        end_match = _end_interview.END_TAG_RE.search(content)
        if end_match:
            self.closing_message = end_match.group(1).strip()
            return None

        # Parse <response>...</response>
        match = re.search(r"<response>(.*?)</response>", content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: strip XML wrapper blocks, return remainder
        content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL)
        content = re.sub(r"<tool_calls>.*?</tool_calls>", "", content, flags=re.DOTALL)
        return content.strip() or None
