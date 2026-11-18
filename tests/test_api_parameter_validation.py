import unittest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.serve_openai_api import app, ChatRequest

class RequestValidationTests(unittest.TestCase):
    def test_invalid_requests_return_422_before_model_access(self):
        client = TestClient(app)
        valid = {'model': 'minimind', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': False}
        cases = ({'messages': []}, {'temperature': -1}, {'temperature': 2.1},
                 {'top_p': -0.1}, {'top_p': 1.1}, {'max_tokens': 0}, {'max_tokens': -3})
        for invalid in cases:
            with self.subTest(invalid=invalid):
                response = client.post('/v1/chat/completions', json={**valid, **invalid})
                self.assertEqual(response.status_code, 422, response.text)

    def test_nonfinite_sampling_values_are_rejected(self):
        for field in ('temperature', 'top_p'):
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    ChatRequest(model='minimind', messages=[{'role': 'user', 'content': 'Hi'}], **{field: value})

    def test_valid_boundary_values_are_accepted(self):
        request = ChatRequest(model='minimind', messages=[{'role': 'user', 'content': 'Hi'}],
                              temperature=2.0, top_p=0.0, max_tokens=1)
        self.assertEqual(request.max_tokens, 1)
