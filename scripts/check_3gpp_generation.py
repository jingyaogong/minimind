"""Run real local generation on fixed DOCX questions; preserve evidence and diagnostics."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_core import ProposalAgent
from rag_evidence import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--base", action="store_true", help="Disable the existing LoRA adapter")
    parser.add_argument("--compare", action="store_true", help="Run both base and LoRA on the same questions")
    parser.add_argument("--output", default="output/agent_generation_smoke.json")
    args = parser.parse_args()
    agent = ProposalAgent()
    runs = []
    questions = [
        "根据 R2-2408133 原文，用三条中文说明主要主张。每条标注来源；不能从片段确定的内容不要补写。",
        "根据 R2-2408133 和 R2-2408134 原文，简要说明两篇提案是否解决同一问题。分别标注来源。",
    ]
    modes = [False, True] if args.compare else [not args.base]
    for use_lora in modes:
        for question in questions:
            print(f"Running {'LoRA' if use_lora else 'base'}: {question}", flush=True)
            events = []
            for event in agent.run_stream(question, max_new_tokens=args.max_tokens, use_lora=use_lora):
                if event["stage"] != "2. STREAMING":
                    events.append(event)
            runs.append({"question": question, "use_lora": use_lora, "events": events})
            terminal = events[-1]
            print(json.dumps({key: terminal.get(key) for key in ("stage", "mode", "result", "validation", "metrics", "msg")}, ensure_ascii=False), flush=True)
    destination = ROOT / args.output
    destination.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(run["events"][-1]["stage"] == "ERROR" for run in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
