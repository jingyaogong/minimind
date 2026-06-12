import ast
import contextlib
import io
import json
import math
import multiprocessing as mp
import os
import re
import statistics
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd


AGENTIC_SYSTEM_PROMPT = (
    "你是一个面向运营数据分析的 Agent。你需要根据用户问题选择合适工具，"
    "优先检索指标口径，再用 Python 对给定 CSV 数据进行计算；最终答案必须简洁，"
    "包含关键结论和必要数值。工具调用必须使用 <tool_call> 包裹的 JSON。"
)


AGENTIC_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retriever",
            "description": "检索指标口径、表字段说明或业务规则，返回最相关的知识片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询"},
                    "top_k": {"type": "integer", "description": "返回片段数量", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_executor",
            "description": "在受限 Python 环境中使用 pandas/numpy 读取给定 CSV 并完成数据分析。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python 代码。把最终结果赋值给 result 或 answer。",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行确定性数学表达式计算，适合简单四则运算、比例和幂运算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 1280 * 0.15 * 3.2"}
                },
                "required": ["expression"],
            },
        },
    },
]


def get_agentic_tools(names=None):
    if names is None:
        names = ["retriever", "python_executor", "calculator"]
    name_set = set(names)
    return [tool for tool in AGENTIC_TOOL_SCHEMAS if tool["function"]["name"] in name_set]


def json_dumps(data):
    return json.dumps(data, ensure_ascii=False, default=str)


def format_agentic_user_prompt(sample):
    table_lines = []
    for table in sample.get("tables", []):
        table_lines.append(
            f"- {table.get('name', '')}: path={table.get('path', '')}; {table.get('description', '')}"
        )
    tool_names = ", ".join(sample.get("tools", ["retriever", "python_executor", "calculator"]))
    return (
        f"用户问题：{sample.get('question', '')}\n\n"
        f"可用数据表：\n" + "\n".join(table_lines) + "\n\n"
        f"可用工具：{tool_names}\n"
        "要求：先确认指标口径，再调用工具计算；最终答案只输出结论，不要编造数据。"
    )


def parse_tool_calls(text):
    calls = []
    for match in re.findall(r"<tool_call>(.*?)</tool_call>", text or "", re.DOTALL):
        raw = match.strip()
        try:
            data = json.loads(raw)
            args = data.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            calls.append(
                {
                    "name": data.get("name", ""),
                    "arguments": args if isinstance(args, dict) else {},
                    "raw": raw,
                    "valid_json": True,
                }
            )
        except Exception:
            calls.append({"name": "", "arguments": {}, "raw": raw, "valid_json": False})
    return calls


def strip_thinking(text):
    text = text or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_final_answer(turn_outputs):
    if isinstance(turn_outputs, str):
        turn_outputs = [turn_outputs]
    if not turn_outputs:
        return ""
    text = strip_thinking(turn_outputs[-1])
    if "</tool_call>" in text:
        text = text.split("</tool_call>")[-1]
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    return text.strip()


def _tokenize(text):
    return re.findall(r"[\w\u4e00-\u9fff]+", str(text).lower())


def _overlap_score(query, text):
    q = Counter(_tokenize(query))
    d = Counter(_tokenize(text))
    return sum(min(q[token], d[token]) for token in q)


def _safe_number(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


class _SafeArithmetic(ast.NodeVisitor):
    allowed_binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }
    allowed_unary = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }
    allowed_funcs = {
        "abs": abs,
        "round": round,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "ceil": math.ceil,
        "floor": math.floor,
        "pow": pow,
    }

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("calculator only supports numeric constants")

    def visit_BinOp(self, node):
        op = type(node.op)
        if op not in self.allowed_binops:
            raise ValueError(f"operator not allowed: {op.__name__}")
        return self.allowed_binops[op](self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node):
        op = type(node.op)
        if op not in self.allowed_unary:
            raise ValueError(f"operator not allowed: {op.__name__}")
        return self.allowed_unary[op](self.visit(node.operand))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in self.allowed_funcs:
            raise ValueError("function not allowed")
        args = [self.visit(arg) for arg in node.args]
        return self.allowed_funcs[node.func.id](*args)

    def generic_visit(self, node):
        raise ValueError(f"node not allowed: {type(node).__name__}")


def safe_calculate(expression):
    normalized = (
        str(expression)
        .replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("（", "(")
        .replace("）", ")")
    )
    tree = ast.parse(normalized, mode="eval")
    return _SafeArithmetic().visit(tree)


def _jsonable(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _validate_python_ast(code):
    banned_nodes = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Lambda, ast.ClassDef)
    banned_names = {
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "os",
        "sys",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "pathlib",
        "shutil",
    }
    banned_attrs = {
        "to_csv",
        "to_excel",
        "to_pickle",
        "to_parquet",
        "write",
        "remove",
        "unlink",
        "rmdir",
        "mkdir",
        "rename",
        "replace",
        "system",
        "popen",
    }
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, banned_nodes):
            raise ValueError(f"python_executor does not allow {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in banned_names:
            raise ValueError(f"name not allowed: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in banned_attrs:
            raise ValueError(f"attribute not allowed: {node.attr}")
    return tree


class _PandasProxy:
    def __init__(self, repo_root, allowed_paths):
        self.repo_root = repo_root
        self.allowed_paths = {os.path.abspath(path) for path in allowed_paths}

    def read_csv(self, path, *args, **kwargs):
        candidates = []
        raw = str(path)
        if os.path.isabs(raw):
            candidates.append(os.path.abspath(raw))
        candidates.append(os.path.abspath(raw))
        candidates.append(os.path.abspath(os.path.join(self.repo_root, raw)))
        allowed_by_basename = {os.path.basename(path): path for path in self.allowed_paths}
        if raw in allowed_by_basename:
            candidates.append(allowed_by_basename[raw])
        for candidate in candidates:
            if candidate in self.allowed_paths:
                return pd.read_csv(candidate, *args, **kwargs)
        raise PermissionError(f"read_csv path is not in sample tables: {raw}")

    def __getattr__(self, name):
        return getattr(pd, name)


def _python_worker(code, repo_root, allowed_paths, queue):
    try:
        tree = _validate_python_ast(code)
        output = io.StringIO()
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "pow": pow,
            "print": print,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }
        local_vars = {}
        global_vars = {
            "__builtins__": safe_builtins,
            "pd": _PandasProxy(repo_root, allowed_paths),
            "np": np,
            "math": math,
            "json": json,
            "statistics": statistics,
        }
        with contextlib.redirect_stdout(output):
            exec(compile(tree, "<agentic_python_executor>", "exec"), global_vars, local_vars)
        value = local_vars.get("result", local_vars.get("answer", output.getvalue().strip()))
        queue.put({"ok": True, "result": _jsonable(value), "error": "", "artifacts": []})
    except Exception as exc:
        queue.put({"ok": False, "result": "", "error": str(exc)[:500], "artifacts": []})


class AgenticToolEnv:
    def __init__(self, sample, repo_root=None, timeout=3):
        self.sample = sample
        self.repo_root = os.path.abspath(repo_root or os.getcwd())
        self.timeout = timeout
        self.allowed_paths = [self._resolve_table_path(table.get("path", "")) for table in sample.get("tables", [])]

    def _resolve_table_path(self, path):
        if os.path.isabs(path):
            return os.path.abspath(path)
        return os.path.abspath(os.path.join(self.repo_root, path))

    def execute(self, name, arguments=None):
        arguments = arguments or {}
        if name == "retriever":
            return self._retriever(arguments)
        if name == "calculator":
            return self._calculator(arguments)
        if name == "python_executor":
            return self._python_executor(arguments)
        return {"ok": False, "result": "", "error": f"unknown tool: {name}", "artifacts": []}

    def _retriever(self, arguments):
        query = str(arguments.get("query", self.sample.get("question", "")))
        top_k = max(1, min(int(arguments.get("top_k", 3) or 3), 8))
        docs = self.sample.get("documents", [])
        scored = []
        for doc in docs:
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            scored.append((_overlap_score(query, text), doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        result = []
        for score, doc in scored[:top_k]:
            result.append(
                {
                    "id": doc.get("id", ""),
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                    "score": score,
                }
            )
        return {"ok": True, "result": result, "error": "", "artifacts": []}

    def _calculator(self, arguments):
        expression = arguments.get("expression", "")
        try:
            value = safe_calculate(expression)
        except Exception as exc:
            return {"ok": False, "result": "", "error": str(exc)[:500], "artifacts": []}
        return {"ok": True, "result": value, "error": "", "artifacts": []}

    def _python_executor(self, arguments):
        code = str(arguments.get("code", ""))
        if not code.strip():
            return {"ok": False, "result": "", "error": "empty code", "artifacts": []}
        queue = mp.Queue(maxsize=1)
        proc = mp.Process(target=_python_worker, args=(code, self.repo_root, self.allowed_paths, queue))
        proc.start()
        proc.join(self.timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return {"ok": False, "result": "", "error": f"python_executor timeout>{self.timeout}s", "artifacts": []}
        if queue.empty():
            return {"ok": False, "result": "", "error": "python_executor returned no result", "artifacts": []}
        return queue.get()


def validate_tool_call(call, available_names):
    if not call.get("valid_json", False):
        return False
    name = call.get("name", "")
    args = call.get("arguments", {})
    if name not in available_names:
        return False
    if name == "retriever":
        return bool(args.get("query"))
    if name == "calculator":
        return bool(args.get("expression"))
    if name == "python_executor":
        return bool(args.get("code"))
    return False


def _f1(pred_items, gold_items):
    pred = set(pred_items)
    gold = set(gold_items)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    if inter == 0:
        return 0.0
    precision = inter / len(pred)
    recall = inter / len(gold)
    return 2 * precision * recall / (precision + recall)


def _numbers(text):
    return [float(x.replace(",", "")) for x in re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", str(text))]


def score_answer(final_answer, sample):
    checks = sample.get("checks", {}) or {}
    score_parts = []
    contains = checks.get("contains") or []
    if contains:
        score_parts.append(float(all(str(item).lower() in final_answer.lower() for item in contains)))
    if "number" in checks:
        target = _safe_number(checks.get("number"))
        tolerance = _safe_number(checks.get("tolerance"), 1e-3)
        nums = _numbers(final_answer)
        score_parts.append(float(any(abs(num - target) <= tolerance for num in nums)))
    if not score_parts and sample.get("answer"):
        score_parts.append(float(str(sample["answer"]).lower() in final_answer.lower()))
    return sum(score_parts) / len(score_parts) if score_parts else 0.0


def repetition_penalty(text, n=3, cap=0.5):
    toks = _tokenize(text)
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def score_agentic_trajectory(turn_outputs, sample, repo_root=None, unfinished=False, execute_tools=True):
    if isinstance(turn_outputs, str):
        turn_outputs = [turn_outputs]
    turn_outputs = turn_outputs or []
    available_names = set(sample.get("tools", ["retriever", "python_executor", "calculator"]))
    expected_tools = sample.get("expected_tools", [])
    env = AgenticToolEnv(sample, repo_root=repo_root)

    calls = []
    tag_errors = 0
    for turn in turn_outputs:
        tag_errors += abs(str(turn).count("<tool_call>") - str(turn).count("</tool_call>"))
        calls.extend(parse_tool_calls(turn))

    valid_flags = [validate_tool_call(call, available_names) for call in calls]
    valid_count = sum(valid_flags)
    invalid_count = len(calls) - valid_count
    selected_tools = [call.get("name", "") for call, ok in zip(calls, valid_flags) if ok]
    tool_f1 = _f1(selected_tools, expected_tools)

    exec_success = 1.0
    if execute_tools and calls:
        successes = []
        for call, ok in zip(calls, valid_flags):
            if not ok:
                successes.append(False)
                continue
            result = env.execute(call["name"], call["arguments"])
            successes.append(bool(result.get("ok")))
        exec_success = sum(successes) / len(successes) if successes else 0.0

    final_answer = extract_final_answer(turn_outputs)
    answer_score = 0.0 if unfinished else score_answer(final_answer, sample)
    task_success = float(answer_score >= 0.999)
    schema_valid_rate = valid_count / len(calls) if calls else 0.0
    invalid_action_rate = invalid_count / len(calls) if calls else 0.0
    format_score = 1.0 if tag_errors == 0 and all(call.get("valid_json") for call in calls) else 0.0
    turn_count = len(turn_outputs)
    expected_turns = max(1, int(sample.get("max_turns", 4)))
    turn_efficiency = max(0.0, 1.0 - max(0, turn_count - expected_turns) / expected_turns)
    repeat_penalty = repetition_penalty(final_answer)

    reward = (
        2.5 * task_success
        + 1.2 * answer_score
        + 0.8 * tool_f1
        + 0.6 * schema_valid_rate
        + 0.6 * exec_success
        + 0.4 * format_score
        + 0.2 * turn_efficiency
        - 0.8 * invalid_action_rate
        - 0.3 * tag_errors
        - repeat_penalty
        - (0.5 if unfinished else 0.0)
    )
    reward = max(min(reward, 5.0), -3.0)
    return reward, {
        "reward_total": reward,
        "task_success": task_success,
        "answer_score": answer_score,
        "tool_selection_f1": tool_f1,
        "schema_valid_rate": schema_valid_rate,
        "exec_success_rate": exec_success,
        "invalid_action_rate": invalid_action_rate,
        "format_score": format_score,
        "turn_efficiency": turn_efficiency,
        "repetition_penalty": repeat_penalty,
        "num_tool_calls": len(calls),
        "avg_turns": turn_count,
    }


def average_agentic_metrics(metrics):
    if not metrics:
        return {}
    keys = sorted({key for row in metrics for key in row})
    return {
        key: sum(float(row.get(key, 0.0)) for row in metrics) / len(metrics)
        for key in keys
    }
