import json
from unittest.mock import MagicMock
from pathlib import Path

from papergym.agents import Paraphraser, PromptLoader


def test_paraphrase_returns_per_domain_dict():
    llm = MagicMock()
    llm.chat.return_value = json.dumps({
        "essence": "process long sequences within bounded compute",
        "paraphrases": {
            "LLM_NLP":    "efficient long-context inference",
            "MULTIMODAL": "efficient cross-modal long fusion",
            "CV":         "long video / high-res image processing",
            "RL":         "credit assignment over long horizons",
            "IR_REC":     "efficient retrieval over very large corpora",
            "SPEECH":     "long-form / streaming audio inference",
            "ROBOTICS":   None,
        }
    })
    prompts = PromptLoader(Path(__file__).parent.parent.parent / "src" /
                           "papergym" / "agents" / "paraphraser")
    result = Paraphraser(llm, prompts).run("long-context efficient inference")
    assert result["essence"].startswith("process long")
    assert result["paraphrases"]["ROBOTICS"] is None
    assert result["paraphrases"]["LLM_NLP"]
    expected_domains = {"LLM_NLP", "MULTIMODAL", "CV", "RL",
                        "IR_REC", "SPEECH", "ROBOTICS"}
    assert set(result["paraphrases"].keys()) == expected_domains
