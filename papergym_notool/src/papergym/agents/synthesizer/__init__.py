from pathlib import Path
from typing import Iterable, Optional

from ..base import BaseAgent, PromptLoader
from ...library import Seed
from ...llm import LLMClient
from ...log import RunLogger


class Synthesizer(BaseAgent):
    def __init__(self, llm: LLMClient, prompts: PromptLoader):
        super().__init__(llm=llm, prompts=prompts, prompt_name="synthesize",
                         temperature=0.5)

    def run(self, query: str, *, seeds: Iterable[Seed],
            events_dir: Optional[Path] = None) -> dict:
        if events_dir is None:
            return self.call(query=query, seeds=list(seeds))
        with RunLogger(events_dir) as logger:
            self.on_message_hook = lambda m: logger.message(
                stage="synthesizer", iter=None, **m)
            try:
                return self.call(query=query, seeds=list(seeds))
            finally:
                self.on_message_hook = None
