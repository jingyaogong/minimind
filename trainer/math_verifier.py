"""Deterministic verifier for numeric math answers.

The verifier operates on decoded model text and deliberately has no torch or
training dependency.  It is therefore usable from data cleaning, evaluation,
and RL reward code without importing the training stack.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Any


ANSWER_MARKER = "答案是"
MAX_ANSWER_LEN = 64

_FULLWIDTH_TRANSLATION = str.maketrans(
    {
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "＋": "+",
        "－": "-",
        "．": ".",
        "，": ",",
        "　": " ",
    }
)
_THOUSANDS_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?\Z")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\Z")


def split_think(response: str | None) -> tuple[str, str, bool]:
    """Split a response into ``thinking``, ``final`` and closure status.

    An unclosed ``<think>`` is treated as malformed rather than allowing an
    answer marker inside the unfinished reasoning to receive reward.
    Responses without any think tag are accepted as final-only responses, but
    ``think_closed`` remains ``False`` so format metrics can distinguish them.
    """

    text = response if isinstance(response, str) else ""
    has_open = "<think>" in text
    has_close = "</think>" in text

    if has_open and not has_close:
        return "", "", False
    if not has_close:
        return "", text.strip(), False

    thinking, final = text.rsplit("</think>", 1)
    if "<think>" in thinking:
        thinking = thinking.split("<think>", 1)[1]
    return thinking.strip(), final.strip(), True


def normalize_number(text: Any) -> str | None:
    """Return a canonical finite decimal string, or ``None`` if invalid.

    The grammar intentionally excludes fractions, units, scientific notation,
    Chinese numerals, and arbitrary comma placement.  Exact decimal strings
    are compared after normalization, so ``59`` equals ``59.0``.
    """

    if not isinstance(text, str):
        return None

    text = text.translate(_FULLWIDTH_TRANSLATION).strip()
    if not text or len(text) > MAX_ANSWER_LEN:
        return None

    if "," in text:
        if not _THOUSANDS_RE.fullmatch(text):
            return None
        text = text.replace(",", "")

    if not _NUMBER_RE.fullmatch(text):
        return None

    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None

    normalized = format(value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"", "+0", "-0"} else normalized


def extract_answer_text(
    response: str | None,
    marker: str = ANSWER_MARKER,
) -> str | None:
    """Extract the first answer line after the last marker in the final text."""

    if not isinstance(marker, str) or not marker:
        raise ValueError("marker must be a non-empty string")

    _, final, _ = split_think(response)
    if marker not in final:
        return None

    tail = final.rsplit(marker, 1)[1].strip()
    if not tail:
        return None
    return tail.splitlines()[0].strip() or None


def verdict(
    response: str | None,
    gt: str | None,
    marker: str = ANSWER_MARKER,
) -> dict[str, Any]:
    """Classify a response as correct, incorrect, no_answer, or invalid_gt."""

    _, final, think_closed = split_think(response)
    extracted = extract_answer_text(response, marker)
    normalized_pred = normalize_number(extracted)
    normalized_gt = normalize_number(gt)

    if normalized_gt is None:
        status = "invalid_gt"
    elif normalized_pred is None:
        status = "no_answer"
    elif normalized_pred == normalized_gt:
        status = "correct"
    else:
        status = "incorrect"

    return {
        "status": status,
        "correct": status == "correct",
        "marker_hit": marker in final,
        "think_closed": think_closed,
        "format_ok": marker in final and think_closed,
        "extracted": extracted,
        "normalized_pred": normalized_pred,
        "normalized_gt": normalized_gt,
    }


def reward_tier(verdict_result: dict[str, Any], mode: str = "binary") -> float:
    """Map a verdict to a deterministic scalar reward."""

    if mode not in {"binary", "tiered"}:
        raise ValueError(f"unsupported reward mode: {mode}")

    status = verdict_result.get("status")
    if status == "correct":
        return 1.0
    if status == "no_answer":
        return -0.5 if mode == "binary" else -1.0
    if status == "incorrect":
        return 0.0 if mode == "binary" else -0.5
    if status == "invalid_gt":
        raise ValueError("invalid ground truth must be removed during data cleaning")
    raise ValueError(f"unknown verdict status: {status}")

