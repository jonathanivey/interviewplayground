from dataclasses import dataclass, field


@dataclass
class Memory:
    content: str = ""
    insight_indices: list[int] = field(default_factory=list)
    reflexive: bool = False
    sensitive: bool = False

    def __post_init__(self):
        # Embedding cache — not persisted to JSON, recomputed on demand.
        self._embedding: list[float] | None = None

    @property
    def is_blank(self) -> bool:
        return self.content == ""

    @property
    def is_insight(self) -> bool:
        return len(self.insight_indices) > 0

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "insight_indices": self.insight_indices,
            "reflexive": self.reflexive,
            "sensitive": self.sensitive,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        m = cls()
        m.content = data["content"]
        m.insight_indices = data["insight_indices"]
        m.reflexive = data["reflexive"]
        m.sensitive = data["sensitive"]
        return m
