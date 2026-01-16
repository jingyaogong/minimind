import re
from typing import Dict, Iterable, List, Set

import ftfy
from langdetect import detect, DetectorFactory
from datasketch import MinHash, MinHashLSH


DetectorFactory.seed = 0


_CONTROL_CHARS = re.compile(r"[\u0000-\u001F\u007F]")
_MULTI_SPACE = re.compile(r"\s{2,}")
_URL = re.compile(r"https?://\S+")


def normalize_text(text: str) -> str:
    """Basic normalization: fix encoding, strip control chars, canonical whitespace."""
    text = ftfy.fix_text(text)
    text = _CONTROL_CHARS.sub(" ", text)
    text = text.replace("\t", " ").replace("\r", " ")
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def is_language(text: str, lang: str = "zh", min_chars: int = 64) -> bool:
    """Heuristic language detection using langdetect.
    Require minimal length to avoid misclassification.
    """
    if len(text) < min_chars:
        return False
    try:
        return detect(text) == lang
    except Exception:
        return False


def filter_text(text: str) -> bool:
    """Simple quality filters: drop extremely short, URL-only, code-only content."""
    if len(text) < 64:
        return False
    if len(text) < 256 and _URL.fullmatch(text):
        return False
    # Drop text with excessive symbols/noise
    noise_ratio = sum(ch in "{}[]<>|~^`" for ch in text) / max(1, len(text))
    if noise_ratio > 0.2:
        return False
    return True


def build_minhash(text: str, num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    # Use 5-gram shingles
    for i in range(max(0, len(text) - 4)):
        shingle = text[i : i + 5]
        mh.update(shingle.encode("utf-8", errors="ignore"))
    return mh


class Deduper:
    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self._keys: Set[str] = set()

    def is_duplicate(self, key: str, text: str) -> bool:
        mh = build_minhash(text, num_perm=self.num_perm)
        if key in self._keys:
            return True
        if self.lsh.query(mh):
            return True
        self.lsh.insert(key, mh)
        self._keys.add(key)
        return False


def clean_record(rec: Dict, target_lang: str = "zh") -> Dict:
    """Clean a single record: normalize, filter, language check."""
    text = normalize_text(rec.get("text", ""))
    if not filter_text(text):
        return {}
    # Prefer records that are in target language, but allow mixed content in Wikipedia
    if not is_language(text, target_lang):
        return {}
    cleaned = {
        "source": rec.get("source"),
        "title": rec.get("title"),
        "text": text,
    }
    return cleaned
