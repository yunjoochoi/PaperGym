from ..base import BaseAgent, PromptLoader
from ...llm import LLMClient


class Paraphraser(BaseAgent):
    def __init__(self, llm: LLMClient, prompts: PromptLoader):
        super().__init__(llm=llm, prompts=prompts, prompt_name="paraphraser",
                         temperature=0.4)

    def run(self, query: str) -> dict:
        return self.call(query=query)
