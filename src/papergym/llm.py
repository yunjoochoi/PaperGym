import os
from dataclasses import dataclass
from typing import Optional

import litellm
from litellm import completion, embedding
from litellm.exceptions import (
    APIError,
    ContextWindowExceededError,
    RateLimitError,
    Timeout,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Drop unsupported params per provider (e.g. temperature for gpt-5) instead of
# erroring. Lets one client speak to many providers without per-model gating.
litellm.drop_params = True


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string from the model


@dataclass
class ChatReply:
    content: Optional[str]
    tool_calls: list[ToolCall]
    raw_message: dict  # provider-agnostic dict, round-trippable in history
    usage: Optional[dict] = None  # {"prompt_tokens": int, "completion_tokens": int}


_RETRYABLE = (RateLimitError, APIError, Timeout)


class LLMClient:
    """Provider-agnostic chat + embeddings via litellm.

    The model argument follows litellm naming: 'gpt-5',
    'claude-sonnet-4-6', 'deepseek/deepseek-chat', 'ollama/llama3', etc.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        truncate_to: int = 10_000,
    ):
        self.model = model or os.environ.get("LITELLM_MODEL")
        self.embedding_model = embedding_model or os.environ.get("EMBEDDING_MODEL")
        # litellm reads provider-specific env (OPENAI_*, ANTHROPIC_*, etc.)
        # itself; we only override when the caller passes them in explicitly.
        self.api_key = api_key
        self.api_base = api_base
        self.truncate_to = truncate_to

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_embedding_tokens = 0

    def _common_kwargs(self, messages: list[dict], temperature: float) -> dict:
        kw: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.api_base:
            kw["api_base"] = self.api_base
        return kw

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=10, max=300),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def _completion(self, **kwargs):
        return completion(**kwargs)

    def _completion_with_truncation(self, **kwargs):
        try:
            return self._completion(**kwargs)
        except ContextWindowExceededError:
            msgs: list[dict] = kwargs["messages"]
            candidates = [i for i, m in enumerate(msgs) if m.get("role") != "system"]
            if not candidates:
                raise
            idx = max(candidates, key=lambda i: len(str(msgs[i].get("content") or "")))
            content = str(msgs[idx].get("content") or "")
            if len(content) <= self.truncate_to:
                raise
            half = self.truncate_to // 2
            msgs[idx]["content"] = (
                content[:half] + "\n...[truncated]...\n" + content[-half:]
            )
            return self._completion(**kwargs)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        response_format: dict | None = None,
    ) -> str:
        kw = self._common_kwargs(messages, temperature)
        if response_format:
            kw["response_format"] = response_format
        resp = self._completion_with_truncation(**kw)
        self.total_prompt_tokens += resp.usage.prompt_tokens
        self.total_completion_tokens += resp.usage.completion_tokens
        return resp.choices[0].message.content

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.7,
        tool_choice: str = "auto",
    ) -> ChatReply:
        """Native tool-calling. Returns parsed ChatReply with tool_calls + content."""
        kw = self._common_kwargs(messages, temperature)
        kw["tools"] = tools
        kw["tool_choice"] = tool_choice
        resp = self._completion_with_truncation(**kw)
        self.total_prompt_tokens += resp.usage.prompt_tokens
        self.total_completion_tokens += resp.usage.completion_tokens
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        raw = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
        return ChatReply(
            content=msg.content,
            tool_calls=calls,
            raw_message=raw,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            },
        )

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=10, max=300),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def embed(self, text: str) -> list[float]:
        kw: dict = {"model": self.embedding_model, "input": text}
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.api_base:
            kw["api_base"] = self.api_base
        resp = embedding(**kw)
        self.total_embedding_tokens += resp.usage.total_tokens
        return resp.data[0]["embedding"]
