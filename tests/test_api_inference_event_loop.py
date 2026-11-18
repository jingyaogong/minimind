import asyncio
import threading
import unittest
from unittest.mock import patch
import httpx
import torch
from scripts import serve_openai_api as server

class Inputs(dict):
    def to(self, device):
        return self

class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    def apply_chat_template(self, *args, **kwargs):
        return 'prompt'
    def __call__(self, *args, **kwargs):
        return Inputs(input_ids=torch.tensor([[3]]), attention_mask=torch.tensor([[1]]))
    def decode(self, *args, **kwargs):
        return 'answer'

class EventLoopResponsivenessTests(unittest.IsolatedAsyncioTestCase):
    async def test_other_coroutines_run_during_nonstream_generation(self):
        release = threading.Event()
        observed = []
        class Model:
            def generate(self, *args, **kwargs):
                observed.append(release.wait(timeout=2))
                return torch.tensor([[3, 4]])
        async def release_from_event_loop():
            await asyncio.sleep(0.05)
            release.set()
        with patch.object(server, 'tokenizer', Tokenizer(), create=True), patch.object(server, 'model', Model(), create=True), patch.object(server, 'device', 'cpu', create=True):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                response, _ = await asyncio.gather(
                    client.post('/v1/chat/completions', json={'model': 'minimind', 'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': False, 'max_tokens': 1}),
                    release_from_event_loop())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['choices'][0]['message']['content'], 'answer')
        self.assertEqual(observed, [True], 'generation blocked the coroutine needed to release it')
