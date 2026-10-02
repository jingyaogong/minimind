"""python -m unittest discover -s tests -p test_stream_tool_calls.py -v"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from openai import OpenAI
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts import serve_openai_api as server


TOOL = '<tool_call>{"name":"weather","arguments":{"city":"杭州"}}</tool_call>'
TOOLS = [{'type': 'function', 'function': {'name': 'weather', 'parameters': {
    'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]


class ScriptedModel:
    def __init__(self, chunks, tokenize=False):
        self.chunks, self.tokenize = chunks, tokenize

    def generate(self, input_ids, streamer, **kwargs):
        if self.tokenize:
            streamer.put(input_ids.cpu())
            for token in streamer.tokenizer.encode(''.join(self.chunks), add_special_tokens=False):
                streamer.put(torch.tensor([token]))
            streamer.end()
        else:
            for chunk in self.chunks:
                streamer.on_finalized_text(chunk)
            streamer.on_finalized_text('', stream_end=True)


class StreamToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            Path(__file__).resolve().parents[1] / 'model', local_files_only=True)

    def setUp(self):
        globals_patch = patch.multiple(server, tokenizer=self.tokenizer, device='cpu',
                                       model=ScriptedModel([]), create=True)
        globals_patch.start()
        self.addCleanup(globals_patch.stop)
        self.http = TestClient(server.app)
        self.addCleanup(self.http.close)
        self.request = {'model': 'minimind', 'messages': [{'role': 'user', 'content': 'Hi'}]}

    def frames(self, chunks, thinking=False):
        server.model = ScriptedModel(chunks)
        response = self.http.post('/v1/chat/completions', json={
            **self.request, 'stream': True, 'open_thinking': thinking})
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/event-stream', response.headers['content-type'])
        return [json.loads(line[6:]) for line in response.text.splitlines()
                if line.startswith('data: ')]

    def assert_output(self, frames, content, count, reasoning=''):
        self.assertFalse(any('error' in frame for frame in frames), frames)
        deltas = [frame['choices'][0]['delta'] for frame in frames]
        self.assertEqual(''.join(d.get('content', '') for d in deltas), content)
        self.assertEqual(''.join(d.get('reasoning_content', '') for d in deltas), reasoning)
        calls = [call for d in deltas for call in d.get('tool_calls', [])]
        self.assertEqual(len(calls), count)
        self.assertEqual([call['index'] for call in calls], list(range(count)))
        self.assertEqual(len({call['id'] for call in calls}), count)
        for call in calls:
            self.assertEqual(call['function']['name'], 'weather')
            self.assertEqual(json.loads(call['function']['arguments']), {'city': '杭州'})
        self.assertEqual(frames[-1]['choices'][0]['finish_reason'], 'tool_calls' if count else 'stop')

    def test_split_markers_and_multiple_calls(self):
        text = 'Before ' + TOOL + ' between ' + TOOL + ' after'
        for chunks in ([text], list(text), ['Before <tool_', text[len('Before <tool_'):]]):
            with self.subTest(chunks=len(chunks)):
                self.assert_output(self.frames(chunks), 'Before  between  after', 2)

    def test_thinking_boundary_and_tool_text_inside_reasoning(self):
        reasoning = 'Consider ' + TOOL + '\n'
        text = reasoning + '</think>\n\n' + TOOL + ' answer'
        self.assert_output(self.frames(list(text), thinking=True), ' answer', 1, reasoning)

    def test_invalid_and_unfinished_calls_preserve_text(self):
        for text in ('plain < text', 'end <tool_', '<tool_call>{"name":',
                     '<tool_call>bad json</tool_call>', '<tool_call>[]</tool_call>'):
            with self.subTest(text=text):
                self.assert_output(self.frames(list(text)), text, 0)
        invalid = '<tool_call>bad json</tool_call>'
        self.assert_output(self.frames([invalid, TOOL]), invalid, 1)

    def test_empty_and_unfinished_thinking(self):
        self.assert_output(self.frames([]), '', 0)
        self.assert_output(self.frames(['reason </thi'], thinking=True), '', 0, 'reason </thi')

    def test_plain_content_and_tool_call_arrive_before_generation_ends(self):
        def generate(input_ids, streamer, **kwargs):
            streamer.on_finalized_text('first ')
            self.assertTrue(continue_generation.wait(5))
            streamer.on_finalized_text(TOOL)
            self.assertTrue(finish_generation.wait(5))
            streamer.on_finalized_text(' last', stream_end=True)

        from threading import Event
        continue_generation, finish_generation = Event(), Event()
        server.model.generate = generate
        stream = server.generate_stream_response(self.request['messages'], 0.7, 0.92, 128)
        self.addCleanup(continue_generation.set)
        self.addCleanup(finish_generation.set)
        first = json.loads(next(stream))
        if first['choices'][0]['delta'].get('role') == 'assistant':
            first = json.loads(next(stream))
        self.assertEqual(first['choices'][0]['delta']['content'], 'first ')
        continue_generation.set()
        call = json.loads(next(stream))
        self.assertEqual(call['choices'][0]['delta']['tool_calls'][0]['function']['name'], 'weather')
        finish_generation.set()
        self.assert_output([first, call] + [json.loads(frame) for frame in stream], 'first  last', 1)

    def test_real_text_streamer_and_openai_client(self):
        server.model = ScriptedModel(['Before ', TOOL, ' after'], tokenize=True)
        with OpenAI(api_key='local-test', base_url='http://testserver/v1',
                    http_client=self.http, max_retries=0) as client:
            chunks = list(client.chat.completions.create(**self.request, tools=TOOLS, stream=True))
        self.assertEqual(''.join(c.choices[0].delta.content or '' for c in chunks), 'Before  after')
        calls = [call for c in chunks for call in c.choices[0].delta.tool_calls or []]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].index, 0)
        self.assertEqual(json.loads(calls[0].function.arguments), {'city': '杭州'})

    def test_tiny_minimind_generation(self):
        torch.manual_seed(42)
        server.model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=64,
            vocab_size=len(self.tokenizer), max_position_embeddings=256)).eval()
        response = self.http.post('/v1/chat/completions', json={
            **self.request, 'stream': True, 'max_tokens': 4})
        frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        self.assertFalse(any('error' in frame for frame in frames), frames)
        self.assertEqual(frames[-1]['choices'][0]['finish_reason'], 'stop')


if __name__ == '__main__':
    unittest.main()
