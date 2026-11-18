import ast
from datetime import datetime, timezone
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts import eval_toolcall

FIXED = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

class FixedDateTime:
    @classmethod
    def now(cls, tz=None):
        return FIXED.astimezone(tz) if tz is not None else FIXED.replace(tzinfo=None)

def web_tool():
    # Loading only the tool function avoids starting Streamlit's page-level UI.
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'web_demo.py'
    module = ast.parse(path.read_text(encoding='utf-8'))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'execute_tool')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['execute_tool']

class CurrentTimeZoneTests(unittest.TestCase):
    def test_eval_tool_uses_requested_zone_and_default(self):
        for zone, expected in (('Asia/Shanghai', '2026-01-15 20:00:00'),
                               ('America/New_York', '2026-01-15 07:00:00'),
                               ('UTC', '2026-01-15 12:00:00')):
            with self.subTest(zone=zone), patch.object(eval_toolcall, 'datetime', FixedDateTime):
                result = eval_toolcall.execute_tool('get_current_time', {'timezone': zone})
                self.assertEqual(result['datetime'], expected)
                self.assertEqual(result['timezone'], zone)
        with patch.object(eval_toolcall, 'datetime', FixedDateTime):
            self.assertEqual(eval_toolcall.execute_tool('get_current_time', {})['datetime'], '2026-01-15 20:00:00')

    def test_web_tool_uses_requested_zone_and_reports_unknown_zone(self):
        execute = web_tool()
        with patch('datetime.datetime', FixedDateTime):
            self.assertEqual(execute('get_current_time', {'timezone': 'Asia/Tokyo'})['result'], '2026-01-15 21:00:00')
        self.assertIn('error', execute('get_current_time', {'timezone': 'Invalid/Timezone'}))
