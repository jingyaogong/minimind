import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class ChatAPIHistoryTests(unittest.TestCase):
    def run_chat(self, history_size, stream):
        path = Path(__file__).resolve().parents[1] / 'scripts' / 'chat_api.py'
        spec = importlib.util.spec_from_file_location('chat_api_under_test', path)
        module = importlib.util.module_from_spec(spec)
        client = Mock()
        requests = []

        def complete(**kwargs):
            requests.append(copy.deepcopy(kwargs['messages']))
            answer = f'answer-{len(requests)}'
            if stream:
                delta = SimpleNamespace(content=answer, reasoning_content=None)
                return [SimpleNamespace(choices=[SimpleNamespace(delta=delta)])]
            message = SimpleNamespace(content=answer)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client.chat.completions.create.side_effect = complete
        with patch('openai.OpenAI', return_value=client):
            spec.loader.exec_module(module)
        module.history_messages_num = history_size
        module.stream = stream
        with patch('builtins.input', side_effect=['question-1', 'question-2', 'question-3', 'question-4', EOFError]), \
                patch('builtins.print'):
            with self.assertRaises(EOFError):
                module.main()
        return requests

    def test_requests_keep_complete_previous_turns(self):
        for stream in (False, True):
            for history_size in (0, 2, 4):
                with self.subTest(stream=stream, history_size=history_size):
                    requests = self.run_chat(history_size, stream)
                    for turn, messages in enumerate(requests, start=1):
                        expected = []
                        first_turn = max(1, turn - history_size // 2)
                        for previous in range(first_turn, turn):
                            expected.extend([
                                {'role': 'user', 'content': f'question-{previous}'},
                                {'role': 'assistant', 'content': f'answer-{previous}'},
                            ])
                        expected.append({'role': 'user', 'content': f'question-{turn}'})
                        self.assertEqual(messages, expected)


if __name__ == '__main__':
    unittest.main()
