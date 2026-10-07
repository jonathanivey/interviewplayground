from __future__ import annotations

import json
import math
import random

from .llm_client import LLMClient
from .memory import Memory
from . import prompts


# JSON Schema for constrained memory-generation decoding (reliable JSON from
# Gemini 3 thinking models). Shared by sequential (Participant) and batch (Study).
MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "reflexive": {"type": "boolean"},
                    "sensitive": {"type": "boolean"},
                },
                "required": ["content", "reflexive", "sensitive"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class Participant:
    def __init__(self):
        self.persona: str = "A typical study participant."
        self.name: str = ""
        self.memories: list[Memory] = []
        self.knowledge: str = "Medium"
        self.verbosity: str = "Medium"
        self.memory: str = "Medium"
        self.reflexivity: str = "Medium"
        self.disclosure: str = "Medium"
        self.understanding: str = "Medium"
        self.transcript: list[dict] = []
        self.retrieval_top_k: int = 5
        self._llm: LLMClient | None = None

    @property
    def memory_size(self) -> int:
        return len(self.memories)

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    def _memory_trait_kwargs(self) -> dict:
        e = prompts.expand_memory_trait
        return {
            "persona": self.persona,
            "knowledge": e("knowledge", self.knowledge),
            "verbosity": e("verbosity", self.verbosity),
            "memory": e("memory", self.memory),
            "reflexivity": e("reflexivity", self.reflexivity),
            "disclosure": e("disclosure", self.disclosure),
            "understanding": e("understanding", self.understanding),
        }

    def _trait_kwargs(self) -> dict:
        e = prompts.expand_trait
        return {
            "persona": self.persona,
            "knowledge": e("knowledge", self.knowledge),
            "verbosity": e("verbosity", self.verbosity),
            "memory": e("memory", self.memory),
            "reflexivity": e("reflexivity", self.reflexivity),
            "disclosure": e("disclosure", self.disclosure),
            "understanding": e("understanding", self.understanding),
        }

    def create_blank_memories(self, n: int) -> None:
        """Append n blank Memory instances. Call before distribute_insights."""
        for _ in range(n):
            self.memories.append(Memory())

    def _build_background_memories_prompt(self) -> tuple[str | None, int]:
        slots = [m for m in self.memories if m.is_blank and not m.is_insight]
        if not slots:
            return None, 0
        n = len(slots)
        prompt = prompts.BACKGROUND_MEMORIES.format(
            n=n,
            reflexive_count=prompts.memory_flag_range("reflexivity", self.reflexivity, n),
            sensitive_count=prompts.memory_flag_range("disclosure", self.disclosure, n),
            **self._memory_trait_kwargs(),
        )
        return prompt, n

    def _apply_background_memories_result(self, generated: list[dict]) -> None:
        slots = [m for m in self.memories if m.is_blank and not m.is_insight]
        for mem, data in zip(slots, generated):
            mem.content = data["content"]
            mem.reflexive = bool(data["reflexive"])
            mem.sensitive = bool(data["sensitive"])

    def _build_insight_memories_prompt(self, insights: list[str]) -> tuple[str | None, int]:
        slots = [m for m in self.memories if m.is_blank and m.is_insight]
        if not slots:
            return None, 0
        topics = []
        for m in slots:
            items = [insights[i] for i in m.insight_indices if i < len(insights)]
            topics.append("; ".join(items))
        topics_list = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(topics))
        n = len(slots)
        prompt = prompts.INSIGHT_MEMORIES.format(
            n=n,
            topics_list=topics_list,
            reflexive_count=prompts.memory_flag_range("reflexivity", self.reflexivity, n),
            sensitive_count=prompts.memory_flag_range("disclosure", self.disclosure, n),
            **self._memory_trait_kwargs(),
        )
        return prompt, n

    def _apply_insight_memories_result(self, generated: list[dict]) -> None:
        slots = [m for m in self.memories if m.is_blank and m.is_insight]
        for mem, data in zip(slots, generated):
            mem.content = data["content"]
            mem.reflexive = bool(data["reflexive"])
            mem.sensitive = bool(data["sensitive"])

    def generate_background_memories(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Fill all blank background memories using a single LLM call."""
        prompt, n = self._build_background_memories_prompt()
        if not n:
            return
        llm = LLMClient(model=model, api_base=api_base, api_key=api_key) if (model or api_base or api_key) else self._get_llm()
        result = llm.call(prompt, json_mode=True, schema=MEMORY_SCHEMA)
        self._apply_background_memories_result(result["memories"])

    def generate_insight_memories(
        self,
        insights: list[str],
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Fill all blank insight memories using a single LLM call."""
        prompt, n = self._build_insight_memories_prompt(insights)
        if not n:
            return
        llm = LLMClient(model=model, api_base=api_base, api_key=api_key) if (model or api_base or api_key) else self._get_llm()
        result = llm.call(prompt, json_mode=True, schema=MEMORY_SCHEMA)
        self._apply_insight_memories_result(result["memories"])

    def ask(
        self,
        question: str,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> str:
        """
        Submit an interviewer question and return the participant's response.

        Adds both the question and response to self.transcript.
        """
        self.transcript.append({"role": "interviewer", "content": question})

        retrieved = self._retrieve_relevant_memories(question)

        if retrieved:
            retrieved_text = "\n".join(
                f"- {m.content}" for m in retrieved
            )
        else:
            retrieved_text = "(no specific memories recalled)"

        participant_label = self.name or "Participant"
        transcript_text = "\n".join(
            f"{'Interviewer' if t['role'] == 'interviewer' else participant_label}: {t['content']}"
            for t in self.transcript[:-1]  # exclude the question just added
        ) or "(beginning of interview)"

        prompt = prompts.ASK.format(
            retrieved_memories=retrieved_text,
            transcript=transcript_text,
            question=question,
            name=participant_label,
            **self._trait_kwargs(),
        )
        llm = LLMClient(model=model, api_base=api_base, api_key=api_key) if (model or api_base or api_key) else self._get_llm()
        response = llm.call(prompt, json_mode=False)

        self.transcript.append({"role": "participant", "content": response})
        return response

    def to_dict(self) -> dict:
        return {
            "persona": self.persona,
            "name": self.name,
            "knowledge": self.knowledge,
            "verbosity": self.verbosity,
            "memory": self.memory,
            "reflexivity": self.reflexivity,
            "disclosure": self.disclosure,
            "understanding": self.understanding,
            "transcript": self.transcript,
            "retrieval_top_k": self.retrieval_top_k,
            "memories": [m.to_dict() for m in self.memories],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Participant":
        p = cls()
        p.persona = data["persona"]
        p.name = data.get("name", "")
        p.knowledge = data["knowledge"]
        p.verbosity = data["verbosity"]
        p.memory = data["memory"]
        p.reflexivity = data["reflexivity"]
        p.disclosure = data["disclosure"]
        p.understanding = data["understanding"]
        p.transcript = data.get("transcript", [])
        p.retrieval_top_k = data.get("retrieval_top_k", 5)
        p.memories = [Memory.from_dict(m) for m in data["memories"]]
        return p

    def save(self, filepath: str) -> None:
        """Save this participant to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "Participant":
        """Load a participant from a JSON file."""
        with open(filepath, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    _MEMORY_SIGMA = {"High": 0.0, "Medium": 0.075, "Low": 0.15}

    def _retrieve_relevant_memories(self, question: str) -> list[Memory]:
        """Return up to retrieval_top_k memories ranked by cosine similarity to question.

        When memory trait is Medium or Low, Gaussian noise is added to scores
        before ranking so that retrieval is less reliable.
        """
        non_blank = [m for m in self.memories if not m.is_blank]
        if not non_blank:
            return []

        llm = self._get_llm()

        # Embed any memories whose embedding hasn't been cached yet (batched).
        unembedded = [m for m in non_blank if m._embedding is None]
        if unembedded:
            vectors = llm.embed([m.content for m in unembedded])
            for m, vec in zip(unembedded, vectors):
                m._embedding = vec

        sigma = self._MEMORY_SIGMA.get(self.memory, 0.0)
        q_vec = llm.embed([question])[0]
        ranked = sorted(
            non_blank,
            key=lambda m: _cosine_similarity(q_vec, m._embedding) + random.gauss(0, sigma),
            reverse=True,
        )
        return ranked[: self.retrieval_top_k]
