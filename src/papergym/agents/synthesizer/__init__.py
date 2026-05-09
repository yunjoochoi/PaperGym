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
            lenses: Optional[Iterable[str]] = None,
            events_dir: Optional[Path] = None) -> dict:
        seeds_list = list(seeds)
        # Each item carries the seed plus the lens text (paraphrase or
        # raw query) that retrieved it; the synthesizer prompt renders
        # the lens so the paraphrase frame is not lost between
        # retrieval and use.
        if lenses is None:
            lenses_list = [""] * len(seeds_list)
        else:
            lenses_list = list(lenses)
        items = [{"seed": s, "lens": l}
                 for s, l in zip(seeds_list, lenses_list)]
        if events_dir is None:
            return self.call(query=query, items=items)
        with RunLogger(events_dir) as logger:
            self.on_message_hook = lambda m: logger.message(
                stage="synthesizer", iter=None, **m)
            try:
                return self.call(query=query, items=items)
            finally:
                self.on_message_hook = None
