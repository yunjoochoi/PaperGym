from enum import Enum


class Domain(Enum):
    LLM_NLP    = "LLM_NLP"
    MULTIMODAL = "MULTIMODAL"
    CV         = "CV"
    RL         = "RL"
    IR_REC     = "IR_REC"
    SPEECH     = "SPEECH"
    ROBOTICS   = "ROBOTICS"


DOMAIN_FIELDS: dict[Domain, list[str]] = {
    Domain.LLM_NLP: [
        "large language model", "language model pretraining",
        "instruction tuning", "in-context learning",
        "chain of thought", "natural language processing",
    ],
    Domain.MULTIMODAL: [
        "vision language model", "multimodal learning",
        "image text pretraining", "vision and language",
    ],
    Domain.CV: [
        "image classification", "object detection",
        "image segmentation", "vision transformer",
        "diffusion model image",
    ],
    Domain.RL: [
        "reinforcement learning", "policy optimization",
        "offline reinforcement learning",
        "reinforcement learning from human feedback",
    ],
    Domain.IR_REC: [
        "dense retrieval", "neural information retrieval",
        "recommender system", "learning to rank",
    ],
    Domain.SPEECH: [
        "speech recognition", "speech synthesis",
        "text to speech", "audio generation",
    ],
    Domain.ROBOTICS: [
        "robot learning", "robotic manipulation",
        "imitation learning robot", "robot policy",
    ],
}

DOMAIN_OVERRIDES: dict[str, Domain] = {
    "Vision and Language":         Domain.MULTIMODAL,
    "Visual Question Answering":   Domain.MULTIMODAL,
    "Image Captioning":            Domain.MULTIMODAL,
}
