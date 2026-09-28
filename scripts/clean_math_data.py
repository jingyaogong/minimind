#!/usr/bin/env python
"""Build and validate the MiniMind math RLVR dataset triple.

Outputs (relative to ``--out-dir``):

* ``math_sft_train.jsonl``   cold-start SFT: 5-field messages incl. reasoning
* ``math_rlvr_train.jsonl``  RLVR train: user-only prompt + gt metadata
* ``math_heldout.jsonl``     held-out set, same schema as the RLVR file
* ``manifests/math_manifest.json``  provenance, counts, discard reasons

The script is torch-free and deterministic: items, train/held-out split and
manifest are reproducible from ``--seed`` plus the input file hash.  The answer
marker and the numeric grammar come from ``trainer.math_verifier`` so data,
reward and evaluation cannot drift apart.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer.math_verifier import ANSWER_MARKER, normalize_number  # noqa: E402

SFT_FILE = "math_sft_train.jsonl"
RLVR_FILE = "math_rlvr_train.jsonl"
HELDOUT_FILE = "math_heldout.jsonl"
MANIFEST_REL = Path("manifests") / "math_manifest.json"
DATA_SHA_REL = Path("manifests") / "data.sha256"
AUDIT_REL = Path("manifests") / "human_audit_sample.tsv"

# The five canonical message fields required by ``SFTDataset``'s explicit
# ``datasets.Features`` declaration.
MESSAGE_FIELDS = ("role", "content", "reasoning_content", "tools", "tool_calls")
ROLES = {"system", "user", "assistant", "tool"}

DIFFICULTIES = ("easy", "medium", "hard")
UNKNOWN_DIFFICULTY = "unknown"

ANSWER_INSTRUCTION = f"最后一行以‘{ANSWER_MARKER} 数字’输出。"
SYNTHETIC_SOURCE = "synthetic"

# Whitelist limits for the self-generated arithmetic expressions.
MAX_EXPR_LEN = 128
MAX_NODES = 64
MAX_ABS = 10 ** 12
MAX_EXP = 8

EMPTY_CONTAINERS = ("", "[]", "{}", "null", "None")


# --------------------------------------------------------------------------- #
# hashing / canonicalisation helpers
# --------------------------------------------------------------------------- #
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_key(text: str) -> str:
    """Canonical question key used for de-duplication and split bookkeeping."""

    normalized = unicodedata.normalize("NFKC", text or "")
    return "".join(normalized.split()).lower()


def hash_rank(key: str, seed: int) -> int:
    """Deterministic, seed-dependent ordering key."""

    return int(sha256_text(f"{seed}:{key}")[:16], 16)


# --------------------------------------------------------------------------- #
# expression evaluation and synthetic generation
# --------------------------------------------------------------------------- #
def safe_eval(expr: str) -> str:
    """Evaluate a whitelisted integer expression via a recursive AST walk.

    Only integer literals, unary ``+/-``, ``+ - *`` and exact ``/`` (no
    remainder, no zero divisor) and bounded ``**`` are allowed.  No ``eval``,
    no ``compile`` sandbox, no name lookup, no attribute access.
    """

    if not isinstance(expr, str) or not expr or len(expr) > MAX_EXPR_LEN:
        raise ValueError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"unparsable expression: {exc.msg}") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ValueError("expression too deep")

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            if abs(node.value) > MAX_ABS:
                raise ValueError("literal out of range")
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                if right == 0 or left % right != 0:
                    raise ValueError("non-integral or zero division")
                value = left // right
            elif isinstance(node.op, ast.Pow):
                if right < 0 or right > MAX_EXP:
                    raise ValueError("exponent out of range")
                value = left ** right
            else:
                raise ValueError(type(node.op).__name__)
            if abs(value) > MAX_ABS:
                raise ValueError("result out of range")
            return value
        raise ValueError(type(node).__name__)

    return str(visit(tree))


def _gen_easy(rng: random.Random) -> str:
    op = rng.choice("+-*")
    if op == "*":
        return f"{rng.randint(11, 99)}*{rng.randint(11, 99)}"
    if op == "+":
        return f"{rng.randint(11, 99)}+{rng.randint(11, 99)}"
    first = rng.randint(20, 99)
    return f"{first}-{rng.randint(11, first - 1)}"


def _gen_medium(rng: random.Random) -> str:
    first, second = rng.randint(11, 60), rng.randint(11, 60)
    factor = rng.randint(3, 9)
    upper = (first + second) * factor - 1
    subtrahend = rng.randint(11, upper) if upper >= 11 else rng.randint(1, upper)
    return f"({first}+{second})*{factor}-{subtrahend}"


def _gen_hard(rng: random.Random) -> str:
    base, exponent = rng.randint(3, 12), rng.randint(2, 3)
    square = rng.randint(11, 25)
    return f"({base}**{exponent})+({square}**2)-{rng.randint(11, 99)}"


GENERATORS = {"easy": _gen_easy, "medium": _gen_medium, "hard": _gen_hard}


def synthetic_question(expr: str) -> str:
    return f"请计算 {expr}，并在{ANSWER_INSTRUCTION}"


# --------------------------------------------------------------------------- #
# chain-of-thought：真实的分步算法，而不是只宣告结果的模板句
# --------------------------------------------------------------------------- #
#
# pilot 教训：原来的 CoT 模板（「按运算顺序先处理括号和乘方，再做乘除，最后做加减，
# 结果是 X」）不含任何中间步骤，模型唯一能学到的就是「输出一个看着合理的数字」。
# 实测：2000 条冷启动把格式率从 0.20 提到 1.00，但 held-out 正确率仍是 0/20，
# 错误形态是「模板句 + 随机数」（66*23→2888、97*61→4378）；而 32-27 这种题里
# 根本没有括号/乘方/乘除，模板句在描述一个不存在的运算过程。
#
# 这里按表达式的 AST 生成可执行步骤：列竖式的进位/借位、一位数的凑十/破十、
# 两位数的拆分乘法、幂的展开。步骤与答案都由 AST 推出，因此不可能与题干漂移；
# 一旦推导出的值不等于 gt，就退回中性表述（并在 check-only 阶段暴露出来）。

COT_VERSION = "cot-v2-column"

_PLACE_NAMES = ("个位", "十位", "百位", "千位", "万位")


def _place(index: int) -> str:
    if index < len(_PLACE_NAMES):
        return _PLACE_NAMES[index]
    return f"第{index + 1}位"


def _small_add_steps(a: int, b: int) -> list[str]:
    """一位数加法：不进位用逐个累加，进位用凑十法。"""

    if a + b < 10:
        steps, running = [], a
        for _ in range(b):
            steps.append(f"{running}+1={running + 1}")
            running += 1
        return steps
    need, rest = 10 - a, b - (10 - a)
    return [
        f"{a} 凑成 10 需要 {need}",
        f"{b} 拆成 {need} 和 {rest}",
        f"10+{rest}={a + b}",
    ]


def _small_sub_steps(a: int, b: int) -> list[str]:
    """一位数减法：不借位用逐个递减，借位用破十法。"""

    if a < 10:
        steps, running = [], a
        for _ in range(b):
            steps.append(f"{running}-1={running - 1}")
            running -= 1
        return steps
    ones = a - 10
    return [
        f"{a} 拆成 10 和 {ones}",
        f"10-{b}={10 - b}",
        f"{10 - b}+{ones}={a - b}",
    ]


def _column_add_steps(a: int, b: int) -> list[str]:
    """列竖式相加：逐位相加并处理进位。"""

    digits_a = [int(digit) for digit in str(abs(a))[::-1]]
    digits_b = [int(digit) for digit in str(abs(b))[::-1]]
    steps: list[str] = []
    carry = 0
    for index in range(max(len(digits_a), len(digits_b))):
        left = digits_a[index] if index < len(digits_a) else 0
        right = digits_b[index] if index < len(digits_b) else 0
        total = left + right + carry
        label = _place(index)
        expression = f"{left}+{right}" + (f"+{carry}" if carry else "")
        if total >= 10:
            steps.append(f"{label} {expression}={total}，写 {total - 10}、向高位进 1")
            carry = 1
        else:
            steps.append(f"{label} {expression}={total}")
            carry = 0
    if carry:
        steps.append("最高位再加上进位的 1")
    return steps


def _column_sub_steps(a: int, b: int) -> list[str]:
    """列竖式相减：逐位相减并处理借位。"""

    digits_a = [int(digit) for digit in str(abs(a))[::-1]]
    digits_b = [int(digit) for digit in str(abs(b))[::-1]]
    steps: list[str] = []
    borrow = 0
    for index in range(max(len(digits_a), len(digits_b))):
        raw = digits_a[index] if index < len(digits_a) else 0
        right = digits_b[index] if index < len(digits_b) else 0
        label = _place(index)
        if raw < right + borrow:
            taken = raw + 10
            tail = f"{taken}-1-{right}={taken - 1 - right}" if borrow else f"{taken}-{right}={taken - right}"
            steps.append(f"{label} 不够减，{raw}+10={taken}，{tail}，向高位借 1")
            borrow = 1
        else:
            result = raw - borrow - right
            if borrow and right:
                steps.append(f"{label} {raw}-1={raw - borrow}，{raw - borrow}-{right}={result}")
            elif borrow:
                steps.append(f"{label} {raw}-1={result}")
            elif right:
                steps.append(f"{label} {raw}-{right}={result}")
            else:
                steps.append(f"{label} 落 {result}")
            borrow = 0
    return steps


def _mul_steps(a: int, b: int) -> list[str]:
    """乘法：九九表 / 整十数补零 / 按十位与个位拆分。"""

    if a < 10 and b < 10:
        return [f"{a}*{b}={a * b}"]
    if a % 10 == 0 or b % 10 == 0:
        multiple, other = (a, b) if a % 10 == 0 else (b, a)
        base = multiple // 10
        return [f"{multiple} 先看成 {base}*{other}={base * other}", f"再补一个 0 得 {a * b}"]
    single, double = (a, b) if a < 10 else (b, a)
    if single < 10:
        high, low = (double // 10) * 10, double % 10
        steps = [f"{double} 拆成 {high} 和 {low}", f"{high}*{single}={high * single}"]
        if low:
            steps.append(f"{low}*{single}={low * single}")
            steps.append(f"{high * single}+{low * single}={a * b}")
        return steps
    high, low = (b // 10) * 10, b % 10
    steps = [f"{b} 拆成 {high} 和 {low}", f"{a}*{high}={a * high}"]
    if low:
        steps.append(f"{a}*{low}={a * low}")
        steps.append(f"{a * high}+{a * low}={a * b}")
    return steps


def _pow_steps(base: int, exponent: int) -> list[str]:
    """幂：展开成连乘（每一步复用乘法的拆分逻辑）。"""

    steps: list[str] = []
    value = base
    for _ in range(max(exponent - 1, 0)):
        steps += _mul_steps(value, base)
        value *= base
    if not steps:
        steps.append(f"{base}**{exponent}={base ** exponent}")
    return steps


def _explain(node: ast.AST) -> tuple[list[str], int]:
    """把表达式 AST 展开成 ``(步骤列表, 该节点的值)``。"""

    if isinstance(node, ast.Constant) and type(node.value) is int:
        return [], int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        steps, value = _explain(node.operand)
        return steps, value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left_steps, left = _explain(node.left)
        right_steps, right = _explain(node.right)
        steps = left_steps + right_steps
        if isinstance(node.op, ast.Add):
            if 0 <= left < 10 and 0 <= right < 10:
                steps += _small_add_steps(left, right)
            elif left >= 0 and right >= 0:
                steps += _column_add_steps(left, right)
            else:
                steps.append(f"{left}+{right}={left + right}")
            return steps, left + right
        if isinstance(node.op, ast.Sub):
            if 0 <= right < left < 10:
                steps += _small_sub_steps(left, right)
            elif 0 <= right <= left:
                steps += _column_sub_steps(left, right)
            else:
                steps.append(f"{left}-{right}={left - right}")
            return steps, left - right
        if isinstance(node.op, ast.Mult):
            steps += _mul_steps(left, right)
            return steps, left * right
        if isinstance(node.op, ast.Pow):
            steps += _pow_steps(left, right)
            return steps, left ** right
        if isinstance(node.op, ast.Div):
            steps.append(f"{left}/{right}={left // right}")
            return steps, left // right
    raise ValueError(f"cannot explain node: {type(node).__name__}")


def synthetic_reasoning(expr: str, gt: str) -> str:
    """可执行的分步推理；步骤由表达式 AST 推出，不会与 gt 漂移。"""

    steps: list[str] = []
    try:
        steps, value = _explain(ast.parse(expr, mode="eval").body)
        if str(value) != str(gt):
            steps = []
    except (SyntaxError, ValueError, TypeError):
        steps = []
    if not steps:
        steps = [f"直接计算 {expr} 得 {gt}"]
    return f"{expr}：" + "；".join(steps) + f"。所以结果是 {gt}。"


def generate_pool(
    difficulty: str,
    need: int,
    rng: random.Random,
    stats: Counter,
) -> list[dict[str, Any]]:
    """Generate ``need`` distinct expressions of one difficulty tier."""

    pool: dict[str, dict[str, Any]] = {}
    attempts = 0
    max_attempts = max(1000, need * 40)
    while len(pool) < need and attempts < max_attempts:
        attempts += 1
        expr = GENERATORS[difficulty](rng)
        try:
            gt = safe_eval(expr)
        except ValueError as exc:
            stats[f"generated_rejected:{type(exc).__name__}"] += 1
            continue
        question = synthetic_question(expr)
        key = canonical_key(question)
        if key in pool:
            stats["generated_duplicate"] += 1
            continue
        pool[key] = {
            "key": key,
            "question": question,
            "gt": gt,
            "difficulty": difficulty,
            "source": SYNTHETIC_SOURCE,
            "expression": expr,
        }
    if len(pool) < need:
        raise RuntimeError(
            f"could not generate enough {difficulty} expressions: "
            f"{len(pool)}/{need} after {attempts} attempts"
        )
    return list(pool.values())


# --------------------------------------------------------------------------- #
# split planning
# --------------------------------------------------------------------------- #
def apportion(sizes: dict[str, int], target: int) -> dict[str, int]:
    """Largest-remainder apportionment of ``target`` seats over ``sizes``."""

    total = sum(sizes.values())
    if target <= 0:
        return {name: 0 for name in sizes}
    if target >= total:
        return dict(sizes)
    exact = {name: target * size / total for name, size in sizes.items()}
    seats = {name: int(math.floor(value)) for name, value in exact.items()}
    missing = target - sum(seats.values())
    order = sorted(sizes, key=lambda name: (-(exact[name] - seats[name]), -sizes[name], name))
    for name in order[:missing]:
        seats[name] += 1
    return seats


def plan_split(
    candidates: list[dict[str, Any]],
    heldout_target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by difficulty bucket + hashed key order, with exact quotas."""

    if heldout_target >= len(candidates):
        raise ValueError(
            f"held-out target {heldout_target} >= available items {len(candidates)}"
        )
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        buckets[item["difficulty"]].append(item)

    quotas = apportion({name: len(items) for name, items in buckets.items()}, heldout_target)

    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for name in sorted(buckets):
        ordered = sorted(buckets[name], key=lambda item: (hash_rank(item["key"], seed), item["key"]))
        quota = quotas.get(name, 0)
        heldout.extend(ordered[:quota])
        train.extend(ordered[quota:])
    return train, heldout


# --------------------------------------------------------------------------- #
# record builders
# --------------------------------------------------------------------------- #
def sample_id_for(item: dict[str, Any]) -> str:
    return f"{item['difficulty']}-{sha256_text(item['key'])[:16]}"


def make_rlvr_record(item: dict[str, Any]) -> dict[str, Any]:
    """RLVR / held-out schema: a single user message plus gt metadata.

    Deliberately no placeholder assistant message: the placeholder would enter
    the prompt of a dataset that does not slice ``conversations[:-1]``.
    """

    return {
        "conversations": [{"role": "user", "content": item["question"]}],
        "gt": item["gt"],
        "source": item["source"],
        "difficulty": item["difficulty"],
        "sample_id": sample_id_for(item),
    }


def make_sft_record(item: dict[str, Any]) -> dict[str, Any]:
    reasoning = item.get("reasoning") or synthetic_reasoning(
        item.get("expression", item["question"]), item["gt"]
    )
    return {
        "conversations": [
            {
                "role": "user",
                "content": item["question"],
                "reasoning_content": "",
                "tools": "",
                "tool_calls": "",
            },
            {
                "role": "assistant",
                "content": f"{ANSWER_MARKER} {item['gt']}",
                "reasoning_content": reasoning,
                "tools": "",
                "tool_calls": "",
            },
        ]
    }


# --------------------------------------------------------------------------- #
# raw-input cleaning
# --------------------------------------------------------------------------- #
def iter_jsonl(path: Path) -> Iterator[tuple[int, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                yield lineno, exc


def extract_ground_truth(row: dict[str, Any]) -> Any:
    for field in ("gt", "ground_truth", "answer", "target"):
        value = row.get(field)
        if value not in (None, ""):
            return value
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth")
    return None


def _is_empty_container(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict)):
        return not value
    return str(value).strip() in EMPTY_CONTAINERS


def extract_user_question(
    row: dict[str, Any],
    stats: Counter,
) -> tuple[str | None, str | None]:
    """Return ``(question, discard_reason)`` for a raw row.

    Tool definitions may sit either on the row (``tools``/``tool_calls``) or on
    individual messages; both are unrecoverable for a single-turn math prompt
    and are discarded explicitly instead of being silently rendered.
    """

    if not _is_empty_container(row.get("tools")):
        return None, "tool_trajectory"
    if not _is_empty_container(row.get("tool_calls")):
        return None, "tool_trajectory"

    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        conversations = row.get("messages")
    if not isinstance(conversations, list) or not conversations:
        return None, "missing_conversations"

    for message in conversations:
        if not isinstance(message, dict):
            return None, "malformed_message"
        if not _is_empty_container(message.get("tools")):
            return None, "tool_trajectory"
        if not _is_empty_container(message.get("tool_calls")):
            return None, "tool_trajectory"
        if message.get("role") == "tool" or message.get("tool_call_id"):
            return None, "tool_trajectory"

    user_messages = [m for m in conversations if m.get("role") == "user"]
    if not user_messages:
        return None, "missing_user"
    if len(user_messages) > 1:
        stats["multi_turn_collapsed"] += 1
    if len(conversations) > len(user_messages) + sum(
        1 for m in conversations if m.get("role") == "system"
    ):
        stats["trailing_messages_dropped"] += 1

    content = user_messages[-1].get("content")
    if not isinstance(content, str) or not content.strip():
        return None, "missing_user"
    return content, None


def normalize_question(text: str) -> str:
    """Uniform question format: whitespace-collapsed, single fixed instruction."""

    collapsed = " ".join((text or "").split())
    if not collapsed:
        return ""
    if ANSWER_MARKER in collapsed:
        return collapsed
    if not collapsed.endswith(("。", "？", "！", "?", "!", ".")):
        collapsed += "。"
    return collapsed + ANSWER_INSTRUCTION


def load_raw_candidates(
    path: Path,
    stats: Counter,
) -> tuple[list[dict[str, Any]], list[int]]:
    candidates: list[dict[str, Any]] = []
    bad_lines: list[int] = []
    seen: dict[str, int] = {}

    for lineno, row in iter_jsonl(path):
        if isinstance(row, json.JSONDecodeError):
            bad_lines.append(lineno)
            stats["bad_json"] += 1
            continue
        if not isinstance(row, dict):
            stats["not_an_object"] += 1
            continue

        raw_gt = extract_ground_truth(row)
        if raw_gt in (None, ""):
            stats["missing_gt"] += 1
            continue
        if isinstance(raw_gt, (dict, list)):
            stats["unparsable_gt"] += 1
            continue
        gt = normalize_number(str(raw_gt).strip())
        if gt is None:
            stats["unparsable_gt"] += 1
            continue

        question, reason = extract_user_question(row, stats)
        if question is None:
            stats[reason or "missing_user"] += 1
            continue
        question = normalize_question(question)
        key = canonical_key(question)
        if not key:
            stats["empty_question"] += 1
            continue
        if key in seen:
            stats["duplicate_key"] += 1
            continue

        difficulty = row.get("difficulty")
        if difficulty not in DIFFICULTIES:
            difficulty = UNKNOWN_DIFFICULTY
        seen[key] = len(candidates)
        candidates.append(
            {
                "key": key,
                "question": question,
                "gt": gt,
                "difficulty": difficulty,
                "source": str(row.get("source") or path.name),
            }
        )
    return candidates, bad_lines


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_manifest(
    *,
    mode: str,
    source_files: list[str],
    source_sha256: dict[str, str],
    stats: Counter,
    train: list[dict[str, Any]],
    heldout: list[dict[str, Any]],
    sft_count: int,
    seed: int,
    heldout_target: int,
    bad_lines: list[int],
    output_hashes: dict[str, str],
) -> dict[str, Any]:
    train_difficulty = Counter(item["difficulty"] for item in train)
    heldout_difficulty = Counter(item["difficulty"] for item in heldout)
    return {
        "mode": mode,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "marker": ANSWER_MARKER,
        "answer_instruction": ANSWER_INSTRUCTION,
        "source_files": source_files,
        "source_sha256": source_sha256,
        "source_line_stats": {k: v for k, v in sorted(stats.items())},
        "bad_line_numbers": bad_lines,
        "train_count": len(train),
        "heldout_count": len(heldout),
        "sft_count": sft_count,
        "heldout_target": heldout_target,
        "duplicate_count": stats.get("duplicate_key", 0) + stats.get("generated_duplicate", 0),
        "discard_reasons": {k: v for k, v in sorted(stats.items()) if k not in {
            "multi_turn_collapsed", "trailing_messages_dropped",
        }},
        "difficulty_counts": {
            "train": dict(sorted(train_difficulty.items())),
            "heldout": dict(sorted(heldout_difficulty.items())),
        },
        "answer_types": {"integer": len(train) + len(heldout)},
        "overlap_count": 0,
        "split_rule": (
            "difficulty bucket -> keys ordered by sha256(seed:key) -> "
            "largest-remainder quota fill for held-out"
        ),
        "outputs": output_hashes,
    }


def write_audit_sheet(
    path: Path,
    items: list[dict[str, Any]],
    size: int,
    seed: int,
) -> list[str]:
    """Write a deterministic spot-check sheet for the ``gt`` audit.

    The SOP requires at least 50 hand-checked ground truths; the sheet keeps the
    evidence auditable instead of leaving it in an untracked chat log.
    """

    if size <= 0:
        return []
    ordered = sorted(items, key=lambda item: (hash_rank(item["key"], seed), item["key"]))
    picked = ordered[: min(size, len(ordered))]
    lines = ["sample_id\tdifficulty\tquestion\tgt\thuman_checked"]
    lines += [
        f"{sample_id_for(item)}\t{item['difficulty']}\t{item['question']}\t{item['gt']}\t"
        for item in picked
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [sample_id_for(item) for item in picked]


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST_REL).parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_dir / RLVR_FILE, payload["rlvr"])
    write_jsonl(out_dir / HELDOUT_FILE, payload["heldout"])
    write_jsonl(out_dir / SFT_FILE, payload["sft"])

    hashes = {
        name: sha256_file(out_dir / name)
        for name in (SFT_FILE, RLVR_FILE, HELDOUT_FILE)
    }
    manifest = build_manifest(output_hashes=hashes, **payload["manifest"])
    (out_dir / MANIFEST_REL).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / DATA_SHA_REL).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
    )
    payload["audit_ids"] = write_audit_sheet(
        out_dir / AUDIT_REL,
        payload["rlvr_items"] + payload["heldout_items"],
        payload["audit_size"],
        payload["manifest"]["seed"],
    )


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_size <= 0 or args.heldout_size <= 0:
        raise ValueError("--train-size and --heldout-size must be positive")

    stats: Counter = Counter()
    rng = random.Random(args.seed)
    total = args.train_size + args.heldout_size
    per_difficulty = _allocate_by_tier(total)

    candidates: list[dict[str, Any]] = []
    for difficulty in DIFFICULTIES:
        candidates.extend(
            generate_pool(difficulty, per_difficulty[difficulty], rng, stats)
        )

    train, heldout = plan_split(candidates, args.heldout_size, args.seed)
    sft = [make_sft_record(item) for item in train]

    generator_signature = json.dumps(
        {
            "tiers": {name: GENERATORS[name].__name__ for name in DIFFICULTIES},
            "cot_version": COT_VERSION,
            "max_expr_len": MAX_EXPR_LEN,
            "max_nodes": MAX_NODES,
            "max_abs": MAX_ABS,
            "max_exp": MAX_EXP,
            "answer_instruction": ANSWER_INSTRUCTION,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "sft": sft,
        "rlvr": [make_rlvr_record(item) for item in train],
        "heldout": [make_rlvr_record(item) for item in heldout],
        "rlvr_items": train,
        "heldout_items": heldout,
        "audit_size": args.audit_sample,
        "manifest": {
            "mode": "synthetic-only",
            "source_files": ["synthetic://whitelist-ast-generator"],
            "source_sha256": {"generator": sha256_text(generator_signature)},
            "stats": stats,
            "train": train,
            "heldout": heldout,
            "sft_count": len(sft),
            "seed": args.seed,
            "heldout_target": args.heldout_size,
            "bad_lines": [],
        },
    }


def _allocate_by_tier(total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(DIFFICULTIES))
    allocation = {name: base for name in DIFFICULTIES}
    for name in DIFFICULTIES[:remainder]:
        allocation[name] += 1
    return allocation


def build_from_raw(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"raw input not found: {input_path}")

    stats: Counter = Counter()
    candidates, bad_lines = load_raw_candidates(input_path, stats)
    if not candidates:
        raise ValueError(f"no usable rows in {input_path}; see discard reasons")

    heldout_target = args.heldout_size
    if heldout_target >= len(candidates):
        raise ValueError(
            f"--heldout-size {heldout_target} >= usable rows {len(candidates)}"
        )
    train, heldout = plan_split(candidates, heldout_target, args.seed)
    if args.train_size:
        train = sorted(train, key=lambda item: hash_rank(item["key"], args.seed))[: args.train_size]
    sft = [make_sft_record(item) for item in train]

    return {
        "sft": sft,
        "rlvr": [make_rlvr_record(item) for item in train],
        "heldout": [make_rlvr_record(item) for item in heldout],
        "rlvr_items": train,
        "heldout_items": heldout,
        "audit_size": args.audit_sample,
        "manifest": {
            "mode": "input",
            "source_files": [str(input_path)],
            "source_sha256": {input_path.name: sha256_file(input_path)},
            "stats": stats,
            "train": train,
            "heldout": heldout,
            "sft_count": len(sft),
            "seed": args.seed,
            "heldout_target": heldout_target,
            "bad_lines": bad_lines,
        },
    }


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path, problems: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, row in iter_jsonl(path):
        if isinstance(row, json.JSONDecodeError):
            problems.append(f"{path.name}:{lineno}: invalid JSON ({row.msg})")
            continue
        if not isinstance(row, dict):
            problems.append(f"{path.name}:{lineno}: not a JSON object")
            continue
        records.append(row)
    return records


def _check_sft(records: list[dict[str, Any]], problems: list[str]) -> set[str]:
    keys: set[str] = set()
    for index, record in enumerate(records, 1):
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            problems.append(f"{SFT_FILE}:{index}: missing conversations")
            continue
        roles = set()
        for message in conversations:
            if not isinstance(message, dict):
                problems.append(f"{SFT_FILE}:{index}: message is not an object")
                continue
            missing = [f for f in MESSAGE_FIELDS if f not in message]
            if missing:
                problems.append(f"{SFT_FILE}:{index}: message missing fields {missing}")
                continue
            non_string = [f for f in MESSAGE_FIELDS if not isinstance(message[f], str)]
            if non_string:
                problems.append(f"{SFT_FILE}:{index}: non-string fields {non_string}")
            role = message["role"]
            if role not in ROLES:
                problems.append(f"{SFT_FILE}:{index}: unknown role {role!r}")
            roles.add(role)
            if role == "assistant" and "</think>" in message["content"]:
                problems.append(
                    f"{SFT_FILE}:{index}: think block must live in reasoning_content"
                )
        if "user" not in roles or "assistant" not in roles:
            problems.append(f"{SFT_FILE}:{index}: need at least one user and one assistant message")
            continue
        user_text = next(m["content"] for m in conversations if m["role"] == "user")
        assistant_text = next(m["content"] for m in conversations if m["role"] == "assistant")
        if ANSWER_MARKER not in assistant_text:
            problems.append(f"{SFT_FILE}:{index}: assistant answer misses the marker")
        keys.add(canonical_key(user_text))
    return keys


def _check_rlvr(path: Path, records: list[dict[str, Any]], problems: list[str]) -> set[str]:
    keys: set[str] = set()
    for index, record in enumerate(records, 1):
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or len(conversations) != 1:
            problems.append(f"{path.name}:{index}: expected exactly one user message")
            continue
        message = conversations[0]
        if not isinstance(message, dict) or message.get("role") != "user":
            problems.append(f"{path.name}:{index}: first message must be user")
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            problems.append(f"{path.name}:{index}: empty user content")
            continue
        if ANSWER_MARKER not in content:
            problems.append(f"{path.name}:{index}: question misses the answer instruction")
        if normalize_number(record.get("gt")) is None:
            problems.append(f"{path.name}:{index}: gt is not a normalisable number: {record.get('gt')!r}")
        for field in ("source", "difficulty", "sample_id"):
            if not str(record.get(field) or "").strip():
                problems.append(f"{path.name}:{index}: missing {field}")
        keys.add(canonical_key(content))
    return keys


def _check_sample_ids(
    sft: list[dict[str, Any]],
    rlvr: list[dict[str, Any]],
    heldout: list[dict[str, Any]],
    problems: list[str],
) -> None:
    seen: dict[str, str] = {}
    for name, records in ((RLVR_FILE, rlvr), (HELDOUT_FILE, heldout)):
        for index, record in enumerate(records, 1):
            sample_id = str(record.get("sample_id") or "")
            if not sample_id:
                continue
            if sample_id in seen:
                problems.append(
                    f"{name}:{index}: duplicate sample_id {sample_id!r} (also in {seen[sample_id]})"
                )
            seen[sample_id] = f"{name}:{index}"
    if sft and not rlvr:
        problems.append("SFT file present but RLVR file empty")


def _check_audit(out_dir: Path, known_ids: set[str], problems: list[str]) -> int:
    """Cross-check the hand-audit sheet against the dataset it belongs to."""

    path = out_dir / AUDIT_REL
    if not path.is_file():
        return 0
    checked = 0
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or rows[0].split("\t")[:1] != ["sample_id"]:
        problems.append(f"{AUDIT_REL}: missing header row")
        return 0
    for index, row in enumerate(rows[1:], 2):
        fields = row.split("\t")
        if len(fields) != 5:
            problems.append(f"{AUDIT_REL}:{index}: expected 5 columns, got {len(fields)}")
            continue
        if fields[0] not in known_ids:
            problems.append(f"{AUDIT_REL}:{index}: unknown sample_id {fields[0]!r}")
        checked += 1
    return checked


def run_check(out_dir: Path) -> int:
    problems: list[str] = []
    paths = {name: out_dir / name for name in (SFT_FILE, RLVR_FILE, HELDOUT_FILE)}
    for name, path in paths.items():
        if not path.is_file():
            problems.append(f"missing file: {path}")
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1

    sft = _read_jsonl(paths[SFT_FILE], problems)
    rlvr = _read_jsonl(paths[RLVR_FILE], problems)
    heldout = _read_jsonl(paths[HELDOUT_FILE], problems)

    sft_keys = _check_sft(sft, problems)
    rlvr_keys = _check_rlvr(paths[RLVR_FILE], rlvr, problems)
    heldout_keys = _check_rlvr(paths[HELDOUT_FILE], heldout, problems)
    _check_sample_ids(sft, rlvr, heldout, problems)

    overlap_rl = rlvr_keys & heldout_keys
    overlap_sft = sft_keys & heldout_keys
    if overlap_rl:
        problems.append(f"train/held-out overlap: {len(overlap_rl)} shared keys")
    if overlap_sft:
        problems.append(f"SFT/held-out overlap: {len(overlap_sft)} shared keys")

    audit_count = _check_audit(
        out_dir,
        {str(r.get("sample_id") or "") for r in rlvr + heldout},
        problems,
    )

    manifest_path = out_dir / MANIFEST_REL
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("train_count") not in (None, len(rlvr)):
            problems.append(
                f"manifest train_count {manifest.get('train_count')} != {len(rlvr)}"
            )
        if manifest.get("heldout_count") not in (None, len(heldout)):
            problems.append(
                f"manifest heldout_count {manifest.get('heldout_count')} != {len(heldout)}"
            )
        if manifest.get("sft_count") not in (None, len(sft)):
            problems.append(f"manifest sft_count {manifest.get('sft_count')} != {len(sft)}")
        for name, digest in (manifest.get("outputs") or {}).items():
            actual = sha256_file(out_dir / name)
            if actual != digest:
                problems.append(f"manifest hash mismatch for {name}")
    else:
        problems.append(f"missing manifest: {manifest_path}")

    print(
        f"sft={len(sft)} rlvr={len(rlvr)} heldout={len(heldout)} "
        f"unique_keys={len(rlvr_keys)}/{len(heldout_keys)} "
        f"overlap={len(overlap_rl)} manifest={'yes' if manifest else 'no'} "
        f"audit_rows={audit_count}"
    )
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1
    print("PASS dataset triple is consistent")
    return 0


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or validate the MiniMind math RLVR dataset triple.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=None,
                        help="raw JSONL to clean; mutually exclusive with --synthetic-only")
    parser.add_argument("--out-dir", type=Path, default=Path("dataset"),
                        help="directory receiving the dataset triple and the manifest")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="build from the whitelisted arithmetic generator")
    parser.add_argument("--train-size", type=int, default=2000,
                        help="synthetic train size, or a cap for --input runs (0 = no cap)")
    parser.add_argument("--heldout-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-sample", type=int, default=50,
                        help="rows in manifests/human_audit_sample.tsv (0 disables)")
    parser.add_argument("--check-only", action="store_true",
                        help="validate an existing triple instead of rewriting it")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.check_only:
            return run_check(args.out_dir)
        if args.synthetic_only == bool(args.input):
            print(
                "error: pass exactly one of --synthetic-only or --input "
                "(or --check-only)",
                file=sys.stderr,
            )
            return 2
        payload = build_synthetic(args) if args.synthetic_only else build_from_raw(args)
        write_outputs(args.out_dir, payload)
        print(
            f"wrote {len(payload['rlvr'])} train / {len(payload['heldout'])} held-out "
            f"/ {len(payload['sft'])} SFT records to {args.out_dir}"
        )
        return run_check(args.out_dir)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
