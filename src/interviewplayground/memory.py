from dataclasses import dataclass, field


@dataclass
class Memory:
    content: str = ""
    target_indices: list[int] = field(default_factory=list)
    reflexive: bool = False
    sensitive: bool = False

    def __post_init__(self):
        # Embedding cache — not persisted to JSON, recomputed on demand.
        self._embedding: list[float] | None = None

    @property
    def is_blank(self) -> bool:
        return self.content == ""

    @property
    def is_target(self) -> bool:
        return len(self.target_indices) > 0

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "target_indices": self.target_indices,
            "reflexive": self.reflexive,
            "sensitive": self.sensitive,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        m = cls()
        m.content = data["content"]
        m.target_indices = data["target_indices"]
        m.reflexive = data["reflexive"]
        m.sensitive = data["sensitive"]
        return m
