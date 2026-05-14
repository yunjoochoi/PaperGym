import json
import time
from pathlib import Path


class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.run_dir / "events.jsonl", "a", encoding="utf-8")

    def event(self, event_type: str, **fields) -> None:
        record = {"ts": time.time(), "type": event_type, **fields}
        self._fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fp.flush()

    def message(self, stage: str, iter, **fields) -> None:
        record = {"ts": time.time(), "type": "message",
                  "stage": stage, "iter": iter, **fields}
        self._fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
