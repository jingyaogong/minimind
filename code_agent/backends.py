"""Inference adapters for the execution-feedback agent."""

from __future__ import annotations

from typing import Any


class OpenAICompatibleGenerator:
    """Callable adapter for MiniMind, vLLM, SGLang, or other OpenAI-style APIs."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8998/v1",
        api_key: str = "not-needed",
        model: str = "minimind",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        open_thinking: bool = False,
        seed: int | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the project 'openai' dependency to use this backend.") from exc
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.open_thinking = open_thinking
        self.seed = seed
        self._request_index = 0

    def __call__(self, prompt: str) -> str:
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "extra_body": {"open_thinking": self.open_thinking},
        }
        if self.seed is not None:
            request["seed"] = self.seed + self._request_index
            self._request_index += 1
        response = self.client.chat.completions.create(
            **request,
        )
        if not response.choices or response.choices[0].message is None:
            raise RuntimeError("Model API returned no completion choice.")
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Model API returned an empty completion.")
        return content
