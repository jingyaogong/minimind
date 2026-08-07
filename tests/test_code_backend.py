from types import SimpleNamespace

import pytest

from code_agent import OpenAICompatibleGenerator


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_openai_backend_builds_non_streaming_request():
    completions = FakeCompletions("```python\ndef solve():\n    return 1\n```")
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    generator = OpenAICompatibleGenerator(
        client=client,
        model="local-minimind",
        temperature=0.2,
        max_tokens=128,
        open_thinking=True,
        seed=17,
    )

    assert generator("solve it").startswith("```python")
    assert completions.kwargs["messages"][0]["content"] == "solve it"
    assert completions.kwargs["stream"] is False
    assert completions.kwargs["extra_body"] == {"open_thinking": True}
    assert completions.kwargs["seed"] == 17

    generator("solve another")
    assert completions.kwargs["seed"] == 18


def test_openai_backend_rejects_empty_completion():
    completions = FakeCompletions("")
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    generator = OpenAICompatibleGenerator(client=client)

    with pytest.raises(RuntimeError, match="empty completion"):
        generator("solve it")
