import uuid
from dataclasses import dataclass, asdict
from ..domain import Domain


def new_seed_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Seed:
    seed_id:     str
    problem:     str
    method:      str
    domain:      Domain
    paper_title: str
    paper_id:    str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["domain"] = self.domain.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Seed":
        d = dict(d)
        d["domain"] = Domain(d["domain"])
        return cls(**d)
