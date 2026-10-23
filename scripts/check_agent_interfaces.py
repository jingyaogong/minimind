"""Check the local API and Streamlit workflow without loading any real model."""

import faulthandler
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
tempfile.tempdir = str(ROOT / "output")
faulthandler.dump_traceback_later(45, exit=True)

from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from server_demo import app, get_agent


def main():
    checks = {}
    with TestClient(app) as client:
        checks["health_model_not_loaded"] = client.get("/health").json() == {"status": "ok", "model_loaded": False}
        checks["invalid_request_rejected"] = client.post("/v1/chat/completions", json={"messages": []}).status_code == 422
        response = client.post("/v1/proposals/analyze", json={
            "question": "对比 R2-2408133 和 R2-2408134", "retrieval_only": True,
        })
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
        last = events[-1]
        checks["analysis_sse_complete"] = response.status_code == 200 and last["stage"] == "4. COMPLETE"
        checks["both_documents_in_ui_data"] = {item["tdoc"] for item in last["sources"]["evidence"]} == {"R2-2408133", "R2-2408134"}
        checks["analysis_done_marker"] = "data: [DONE]" in response.text

        class FakeRuntime:
            model = None
            def stream(self, *args, **kwargs):
                yield "test answer"

        real_runtime = get_agent().runtime
        get_agent().runtime = FakeRuntime()
        try:
            response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}], "stream": False})
            checks["nonstream_chat"] = response.json()["choices"][0]["message"]["content"] == "test answer"
            response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
            checks["stream_chat"] = '"content": "test answer"' in response.text and "[DONE]" in response.text
        finally:
            get_agent().runtime = real_runtime

    ui = AppTest.from_file(str(ROOT / "app_web.py")).run(timeout=20)
    checks["initial_ui_no_exception"] = len(ui.exception) == 0
    ui.text_area[0].set_value("对比 R2-2408133 和 R2-2408134")
    next(button for button in ui.button if button.label == "检索原文").click().run(timeout=20)
    checks["ui_retrieval_complete"] = len(ui.exception) == 0 and "last_run" in ui.session_state
    if checks["ui_retrieval_complete"]:
        checks["ui_has_downloads"] = ui.session_state["last_run"]["mode"] == "retrieval_only"
    destination = ROOT / "output/agent_interface_checks.json"
    destination.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    faulthandler.cancel_dump_traceback_later()
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
