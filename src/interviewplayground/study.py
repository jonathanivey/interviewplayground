from __future__ import annotations

import json
import random
import warnings

from .llm_client import LLMClient
from .participant import Participant, MEMORY_SCHEMA
from . import prompts
from .interviewreportcard import (
    evaluate_participant_responses,
    evaluate_interviewer_behavior,
    evaluate_participant_experience,
    evaluate_conversation_length,
)


class Study:
    def __init__(
        self,
        target_information: list[str],
        research_questions: list[str] | None = None,
        interview_guide: list[dict] | None = None,
    ):
        self.target_information: list[str] = target_information
        self.research_questions: list[str] = research_questions or []
        self.interview_guide: list[dict] = interview_guide or []
        self.participants: list[Participant] = []

    @property
    def n(self) -> int:
        return len(self.participants)

    @classmethod
    def from_description(
        cls,
        description: str,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> "Study":
        """Generate target_information from a qualitative study description using an LLM."""
        llm = LLMClient(model=model, api_base=api_base, api_key=api_key)
        prompt = prompts.STUDY_FROM_DESCRIPTION.format(description=description)
        result = llm.call(prompt, json_mode=True)
        return cls(target_information=result["target_information"])

    def save(self, filepath: str) -> None:
        """Save this study to a JSON file."""
        data = {
            "target_information": self.target_information,
            "research_questions": self.research_questions,
            "interview_guide": self.interview_guide,
            "participants": [p.to_dict() for p in self.participants],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "Study":
        """Load a study from a JSON file."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        study = cls(
            target_information=data["target_information"],
            research_questions=data.get("research_questions", []),
            interview_guide=data.get("interview_guide", []),
        )
        study.participants = [Participant.from_dict(p) for p in data["participants"]]
        return study

    def add_participant(self, participant: Participant) -> None:
        """Add a configured Participant to this study."""
        self.participants.append(participant)

    def remove_participant(self, k: int) -> Participant:
        """Remove and return the participant at index k."""
        return self.participants.pop(k)

    def create_participants(self, n: int) -> None:
        """Append n default Participant instances with Medium traits and no memories."""
        for _ in range(n):
            self.participants.append(Participant())

    def distribute_target_information(
        self,
        avg_per_participant: float,
        ensure_all_distributed: bool = True,
    ) -> None:
        """
        Assign target_indices to blank memories across all participants.

        Each participant must have already called create_blank_memories().
        avg_per_participant controls how many target items each participant
        knows about on average. If ensure_all_distributed is True, every
        target item will appear in at least one participant's memories.

        Raises ValueError if a participant doesn't have enough blank memories
        to accommodate their assigned target items.
        """
        if not self.participants:
            return

        T = len(self.target_information)
        if T == 0:
            return

        target_count = max(1, min(T, round(avg_per_participant)))

        # Step 1: assign a set of target indices to each participant
        assignments: list[list[int]] = []
        for _ in self.participants:
            indices = random.sample(range(T), target_count)
            assignments.append(indices)

        # Step 2: gap-fill so every target index is covered
        if ensure_all_distributed:
            covered = set(idx for a in assignments for idx in a)
            missing = set(range(T)) - covered
            for idx in missing:
                # give it to the participant with the fewest targets so far
                least = min(range(len(assignments)), key=lambda i: len(assignments[i]))
                assignments[least].append(idx)

        # Step 3: assign indices to blank memory slots
        for participant, target_indices in zip(self.participants, assignments):
            blank = [m for m in participant.memories if m.is_blank]
            needed = len(target_indices)
            if len(blank) < needed:
                raise ValueError(
                    f"Participant has {len(blank)} blank memories but needs "
                    f"{needed} target slots. Call create_blank_memories() with "
                    f"a larger value first."
                )
            random.shuffle(blank)
            for slot, target_idx in zip(blank, target_indices):
                slot.target_indices = [target_idx]

    def generate_all_memories(
        self,
        target_info: list[str] | None = None,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Generate memories for all participants in a single batch LLM call.

        Collects nontarget and target memory prompts from every participant,
        submits them as one batch via LLMClient.batch_call(), then routes
        results back. Requires a batch-compatible model (openai, azure, gemini).

        target_info defaults to self.target_information when not provided.
        """
        prompts_list, routing = self._build_memory_prompts(target_info)

        if not prompts_list:
            return

        llm = LLMClient(model=model, api_base=api_base, api_key=api_key)
        results = llm.batch_call(prompts_list, json_mode=True, schema=MEMORY_SCHEMA)
        self._apply_memory_results(results, routing)

    def _build_memory_prompts(
        self,
        target_info: list[str] | None = None,
    ) -> tuple[list[str], list[tuple[str, int]]]:
        """Collect memory prompts and their routing across all participants."""
        resolved_target_info = target_info if target_info is not None else self.target_information

        prompts_list: list[str] = []
        routing: list[tuple[str, int]] = []

        for i, p in enumerate(self.participants):
            prompt, n = p._build_nontarget_memories_prompt()
            if n > 0:
                prompts_list.append(prompt)
                routing.append(("nontarget", i))
            prompt, n = p._build_target_memories_prompt(resolved_target_info)
            if n > 0:
                prompts_list.append(prompt)
                routing.append(("target", i))

        return prompts_list, routing

    def _apply_memory_results(
        self,
        results: list[dict],
        routing: list[tuple[str, int]],
    ) -> None:
        """Route batch results back to the participants that produced them."""
        for result, (kind, p_idx) in zip(results, routing):
            p = self.participants[p_idx]
            if kind == "nontarget":
                p._apply_nontarget_memories_result(result["memories"])
            else:
                p._apply_target_memories_result(result["memories"])

    def submit_all_memories(
        self,
        target_info: list[str] | None = None,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> dict:
        """
        Submit memory generation as a batch job and return a JSON-serializable
        handle, without waiting for completion. Pass the handle to
        retrieve_all_memories() later (e.g. the next day) to fetch the results
        and apply them to the participants.

        NOTE: retrieval applies results by participant index, so the same Study
        object (with participants in the same order) must be used to retrieve.
        Returns an empty handle (no batch_id) if there are no memories to build.
        """
        prompts_list, routing = self._build_memory_prompts(target_info)
        if not prompts_list:
            return {"function": "generate_all_memories", "batch_id": None, "routing": []}

        llm = LLMClient(model=model, api_base=api_base, api_key=api_key)
        batch_id = llm.batch_submit(prompts_list, json_mode=True, schema=MEMORY_SCHEMA)
        return {
            "function": "generate_all_memories",
            "batch_id": batch_id,
            "model": llm.model,
            "routing": routing,
        }

    def retrieve_all_memories(
        self,
        handle: dict,
        api_key: str | None = None,
    ) -> None:
        """
        Retrieve a memory batch previously submitted with submit_all_memories()
        and apply the results to this Study's participants. Raises RuntimeError
        if the batch is not yet complete.

        The Study's participants must be in the same order as at submit time
        (the handle routes results by participant index).
        """
        if not handle.get("batch_id"):
            return
        llm = LLMClient(model=handle["model"], api_key=api_key)
        results = llm.batch_retrieve(handle["batch_id"], json_mode=True)
        # routing tuples round-trip through JSON as lists — normalize back.
        routing = [(kind, idx) for kind, idx in handle["routing"]]
        self._apply_memory_results(results, routing)

    def evaluate(
        self,
        interview_guide: list[dict] | None = None,
        research_questions: list[str] | None = None,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> dict:
        """
        Evaluate interview quality with the InterviewReportCard suite.

        Collects transcripts from all participants, then computes one
        descriptive bucket plus three InterviewReportCard dimensions, each an
        LLM-judged set of metrics:

          conversation_length (no LLM calls, computed directly from
            transcripts): avg_turns, avg_response_length.

          participant_responses: relevant_response_volume,
            interview_guide_coverage, novel_responses.

          interviewer_behavior: coherence, adaptiveness, leading_questions,
            support_rapport, unclear_questions.

          participant_experience: comfort_level, overall_experience.

        Every metric in every dimension is a per-participant average; each
        dimension's returned dict also carries a "per_participant" list with
        each participant's own unaggregated value(s).

        Args:
            interview_guide: The guide used during interviews. Each item is a
                dict with "topic" (str) and "subtopics" (list[str]) keys.
                Defaults to self.interview_guide.
            research_questions: List of research question strings to evaluate
                participant responses against. Defaults to self.research_questions.
            model, api_base, api_key: Optional LLM overrides forwarded to the
                three LLM-judged evaluation calls (conversation_length makes no
                LLM calls, so these don't apply to it).

        Returns:
            {"conversation_length": {...}, "participant_responses": {...},
             "interviewer_behavior": {...}, "participant_experience": {...}}
        """
        interview_guide = interview_guide if interview_guide is not None else self.interview_guide
        research_questions = research_questions if research_questions is not None else self.research_questions
        if not self.participants:
            warnings.warn(
                "Study.evaluate: study has no participants — returning zero values.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "conversation_length": {"avg_turns": 0.0, "avg_response_length": 0.0,
                                        "per_participant": []},
                "participant_responses": {"relevant_response_volume": 0.0,
                                          "interview_guide_coverage": 0.0,
                                          "novel_responses": 0.0, "per_participant": []},
                "interviewer_behavior": {"coherence": 0.0, "adaptiveness": 0.0,
                                         "leading_questions": 0.0, "support_rapport": 0.0,
                                         "unclear_questions": 0.0, "per_participant": []},
                "participant_experience": {"comfort_level": 0.0,
                                           "overall_experience": 0.0, "per_participant": []},
            }

        transcripts = [p.transcript for p in self.participants]
        kwargs = dict(model=model, api_base=api_base, api_key=api_key)

        return {
            "conversation_length": evaluate_conversation_length(
                transcripts=transcripts,
            ),
            "participant_responses": evaluate_participant_responses(
                transcripts=transcripts,
                research_questions=research_questions,
                interview_guide=interview_guide,
                **kwargs,
            ),
            "interviewer_behavior": evaluate_interviewer_behavior(
                transcripts=transcripts,
                **kwargs,
            ),
            "participant_experience": evaluate_participant_experience(
                transcripts=transcripts,
                **kwargs,
            ),
        }
