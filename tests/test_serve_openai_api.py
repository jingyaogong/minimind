"""Run with: python -m unittest discover -s tests -p 'test_serve_openai_api.py' -v.

Uses the repository tokenizer, real TextStreamer, FastAPI's HTTP test transport,
and the OpenAI SDK. No downloaded weights, API credentials or GPU are needed.
"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from openai import APIError, OpenAI
from openai.types.chat import ChatCompletionChunk
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts import serve_openai_api as server


class ScriptedModel:
    """Send deterministic token IDs through the actual TextStreamer."""

    def __init__(self, tokenizer, text="Hello world!", error=False):
        self.tokenizer, self.text, self.error = tokenizer, text, error

    def generate(self, input_ids, streamer=None, **kwargs):
        tokens = self.tokenizer.encode(self.text, add_special_tokens=False)
        if streamer is not None:
            streamer.put(input_ids.cpu())
            for token in tokens:
                streamer.put(torch.tensor([token]))
            if self.error:
                raise RuntimeError("synthetic generation failure")
            streamer.end()
        return torch.cat([input_ids, input_ids.new_tensor([tokens])], dim=1)


class StreamingAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            Path(__file__).resolve().parents[1] / "model", local_files_only=True
        )

    def setUp(self):
        globals_patch = patch.multiple(
            server, tokenizer=self.tokenizer, device="cpu",
            model=ScriptedModel(self.tokenizer), create=True
        )
        globals_patch.start()
        self.addCleanup(globals_patch.stop)
        self.http = TestClient(server.app)
        self.sdk = OpenAI(
            api_key="local-test-key", base_url="http://testserver/v1",
            http_client=self.http, max_retries=0, _strict_response_validation=True
        )
        self.addCleanup(self.sdk.close)
        self.request = {"model": "minimind", "messages": [{"role": "user", "content": "Hi"}]}

    def frames(self, **extra):
        response = self.http.post("/v1/chat/completions", json={**self.request, "stream": True, **extra})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        return [json.loads(line[6:]) for line in response.text.splitlines()
                if line.startswith("data: ") and line[6:] != "[DONE]"]

    def sdk_chunks(self, **extra):
        return list(self.sdk.chat.completions.create(**self.request, stream=True, **extra))

    def test_http_chunks_validate_against_sdk_schema(self):
        frames = self.frames()
        for frame in frames:
            chunk = ChatCompletionChunk.model_validate(frame)
            self.assertEqual(chunk.object, "chat.completion.chunk")
            self.assertEqual(chunk.model, "minimind")
            self.assertEqual(chunk.choices[0].index, 0)
        self.assertEqual(len({frame["id"] for frame in frames}), 1)
        self.assertEqual(len({frame["created"] for frame in frames}), 1)
        self.assertNotEqual(frames[0]["id"], self.frames()[0]["id"])

    def test_sdk_reads_plain_content(self):
        chunks = self.sdk_chunks()
        self.assertEqual("".join(c.choices[0].delta.content or "" for c in chunks), "Hello world!")
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")

    def test_sdk_accumulates_assistant_message(self):
        with self.sdk.beta.chat.completions.stream(**self.request) as stream:
            for _ in stream:
                pass
            message = stream.get_final_completion().choices[0].message
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "Hello world!")

    def test_reasoning_and_content_are_preserved(self):
        server.model = ScriptedModel(self.tokenizer, "正在思考\n</think>\n答案")
        chunks = self.sdk_chunks(extra_body={"open_thinking": True})
        self.assertEqual("".join(getattr(c.choices[0].delta, "reasoning_content", "") or "" for c in chunks), "正在思考\n")
        self.assertEqual("".join(c.choices[0].delta.content or "" for c in chunks), "答案")

    def test_sdk_accumulates_multiple_tool_calls(self):
        server.model = ScriptedModel(self.tokenizer,
            '<tool_call>{"name":"weather","arguments":{"city":"杭州"}}</tool_call>\n'
            '<tool_call>{"name":"clock","arguments":{}}</tool_call>')
        chunks = self.sdk_chunks()
        calls = [call for chunk in chunks for call in chunk.choices[0].delta.tool_calls or []]
        self.assertEqual([call.index for call in calls], [0, 1])
        self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")
        with self.sdk.beta.chat.completions.stream(**self.request) as stream:
            for _ in stream:
                pass
            calls = stream.get_final_completion().choices[0].message.tool_calls
        self.assertEqual([call.function.name for call in calls], ["weather", "clock"])
        self.assertEqual(json.loads(calls[0].function.arguments), {"city": "杭州"})
        self.assertEqual(json.loads(calls[1].function.arguments), {})

    def test_generation_error_never_reports_success(self):
        for text in ("", "partial output "):
            with self.subTest(text=text):
                server.model = ScriptedModel(self.tokenizer, text, error=True)
                frames = self.frames()
                self.assertEqual(frames[-1], {"error": "synthetic generation failure"})
                self.assertFalse(any(c.get("finish_reason") for frame in frames for c in frame.get("choices", [])))
                with self.assertRaises(APIError):
                    self.sdk_chunks()

    def test_empty_completion_still_has_valid_terminal_chunk(self):
        server.model = ScriptedModel(self.tokenizer, "")
        chunks = self.sdk_chunks()
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")
        self.assertEqual("".join(c.choices[0].delta.content or "" for c in chunks), "")

    def test_nonstream_response_is_unchanged(self):
        response = self.sdk.chat.completions.create(**self.request, stream=False)
        self.assertEqual(response.choices[0].message.role, "assistant")
        self.assertEqual(response.choices[0].message.content, "Hello world!")
        self.assertEqual(response.choices[0].finish_reason, "stop")

    def test_tiny_minimind_inference_through_http(self):
        torch.manual_seed(42)
        config = MiniMindConfig(
            hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=64,
            vocab_size=len(self.tokenizer), max_position_embeddings=256
        )
        server.model = MiniMindForCausalLM(config).eval()
        chunks = self.sdk_chunks(max_tokens=2)
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")
        self.assertTrue(all(c.id == chunks[0].id for c in chunks))


if __name__ == "__main__":
    unittest.main()
