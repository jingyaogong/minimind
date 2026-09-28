"""Dataset for the math RLVR / GRPO training file.

The input is ``math_rlvr_train.jsonl`` as produced by
``scripts/clean_math_data.py``: one user message plus top-level ``gt``,
``source``, ``difficulty`` and ``sample_id``.

Two deliberate differences from :class:`dataset.lm_dataset.RLAIFDataset`:

* it does **not** slice ``conversations[:-1]``.  The RLVR file has no
  placeholder assistant turn, so that slice would silently delete the only
  user message.
* a file that still carries an assistant turn (for example
  ``math_sft_train.jsonl``) is rejected instead of being trained on.  Pointing
  ``--rlvr_data_path`` at the SFT file would otherwise reward the policy for
  reproducing a gold answer that is already inside its own prompt.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import Dataset  # noqa: E402

from trainer.math_verifier import ANSWER_MARKER, normalize_number  # noqa: E402


def _percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def _length_stats(lengths: list[int], budget: int) -> dict[str, Any]:
    ordered = sorted(lengths)
    overlong = sum(1 for value in lengths if value > budget)
    return {
        "count": len(lengths),
        "min_tokens": ordered[0] if ordered else 0,
        "p50_tokens": _percentile(ordered, 0.50),
        "p95_tokens": _percentile(ordered, 0.95),
        "max_tokens": ordered[-1] if ordered else 0,
        "budget_tokens": budget,
        "overlong": overlong,
        "overlong_fraction": overlong / len(lengths) if lengths else 0.0,
    }


class MathRLDataset(Dataset):
    """User-only math prompts with a verifiable ground-truth answer."""

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        thinking_ratio: float = 1.0,
        max_length: int = 768,
        marker: str = ANSWER_MARKER,
    ) -> None:
        if not 0.0 <= thinking_ratio <= 1.0:
            raise ValueError(f"thinking_ratio must be in [0, 1], got {thinking_ratio}")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        self.path = str(path)
        self.tokenizer = tokenizer
        self.thinking_ratio = thinking_ratio
        # Interface-compatibility field.  Truncation still happens in the
        # training loop; `audit_lengths()` is what makes it visible.
        self.max_length = max_length
        self.marker = marker
        self.samples = self._load()
        if not self.samples:
            raise ValueError(f"no usable samples in {self.path}")

    # ------------------------------------------------------------------ #
    # loading / validation
    # ------------------------------------------------------------------ #
    def _load(self) -> list[dict[str, Any]]:
        path = Path(self.path)
        if not path.is_file():
            raise FileNotFoundError(f"RLVR data not found: {path}")
        samples: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
                samples.append(self._validate(row, lineno, path))
        return samples

    def _validate(self, row: Any, lineno: int, path: Path) -> dict[str, Any]:
        where = f"{path}:{lineno}"
        if not isinstance(row, dict):
            raise ValueError(f"{where}: expected a JSON object")

        conversations = row.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(f"{where}: missing conversations")

        roles = []
        for message in conversations:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError(f"{where}: every message needs a string content")
            roles.append(message.get("role"))
        if any(role != "user" for role in roles):
            raise ValueError(
                f"{where}: expected user-only messages, got roles {roles}. "
                "An assistant turn means this is an SFT file (or a file with a "
                "placeholder answer); the RLVR prompt must not contain the gold "
                "answer. Rebuild it with `scripts/clean_math_data.py`."
            )
        if len(conversations) != 1:
            raise ValueError(
                f"{where}: expected exactly one user message, got {len(conversations)}"
            )

        content = conversations[0]["content"]
        if not content.strip():
            raise ValueError(f"{where}: empty user content")
        if self.marker not in content:
            raise ValueError(
                f"{where}: question does not request the answer marker "
                f"{self.marker!r}; data and verifier marker have drifted apart"
            )

        gt = row.get("gt")
        if gt is None:
            raise ValueError(f"{where}: missing gt")
        normalized_gt = normalize_number(gt if isinstance(gt, str) else str(gt))
        if normalized_gt is None:
            raise ValueError(f"{where}: gt is not a normalisable number: {gt!r}")

        return {
            "conversations": [dict(conversations[0])],
            "gt": normalized_gt,
            "difficulty": str(row.get("difficulty") or "unknown"),
            "source": str(row.get("source") or "unknown"),
            "sample_id": str(row.get("sample_id") or f"{path.name}:{lineno}"),
        }

    # ------------------------------------------------------------------ #
    # dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def render_prompt(self, index: int, open_thinking: bool | None = None) -> str:
        """Render one prompt; ``open_thinking=None`` samples the ratio."""

        if open_thinking is None:
            open_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            self.samples[index]["conversations"],
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=open_thinking,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "prompt": self.render_prompt(index),
            "gt": sample["gt"],
            "difficulty": sample["difficulty"],
            "sample_id": sample["sample_id"],
        }

    # ------------------------------------------------------------------ #
    # auditing
    # ------------------------------------------------------------------ #
    def audit_lengths(self, sample_size: int | None = None) -> dict[str, Any]:
        """Tokenize every rendered prompt once and report length statistics.

        ``__getitem__`` cannot count overlong prompts reliably: with
        ``num_workers > 0`` the counter lives in the worker processes and never
        reaches the parent.  The trainer therefore calls this once at startup
        and records the result, so truncation in the training loop is visible
        instead of silent.
        """

        count = len(self.samples) if sample_size is None else min(sample_size, len(self.samples))
        stats: dict[str, Any] = {"rows": count, "thinking_ratio": self.thinking_ratio}
        for label, open_thinking in (("thinking_on", True), ("thinking_off", False)):
            lengths = [
                len(
                    self.tokenizer(
                        self.render_prompt(index, open_thinking=open_thinking),
                        add_special_tokens=False,
                    ).input_ids
                )
                for index in range(count)
            ]
            stats[label] = _length_stats(lengths, self.max_length)
        return stats

    def difficulty_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in self.samples:
            counts[sample["difficulty"]] = counts.get(sample["difficulty"], 0) + 1
        return dict(sorted(counts.items()))
