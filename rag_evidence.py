"""Local TDoc retrieval with traceable Word passages; no model or training data required."""

from __future__ import annotations

import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent
TDOC_PATTERN = re.compile(r"\bR([1-4])\s*[-\u2010-\u2015]\s*(\d{6,7})\b", re.I)
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
STOP_WORDS = set("the a an of on for and or to in is are with this that discussion please compare explain".split())
QUERY_ALIASES = {
    "非地面": "NTN", "卫星": "NTN satellite", "节能": "energy saving NES",
    "移动性": "mobility LTM", "切换": "handover LTM", "速率控制": "rate control",
    "寻呼": "paging", "逻辑信道": "logical channel LCP", "时延": "delay latency",
}


def extract_tdoc_ids(text: str) -> list[str]:
    """Normalize identifiers and preserve their first-occurrence order."""
    return list(dict.fromkeys(f"R{group}-{number}" for group, number in TDOC_PATTERN.findall(text)))


def tokenize(text: str, expand_query: bool = False) -> list[str]:
    """Use English words and Chinese bigrams, with explicit query aliases."""
    if expand_query:
        text += " " + " ".join(value for key, value in QUERY_ALIASES.items() if key in text)
    words = [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP_WORDS]
    for phrase in re.findall(r"[\u4e00-\u9fff]+", text):
        words.extend(phrase[i:i + 2] for i in range(len(phrase) - 1))
    return words


class BM25Index:
    """Small inverted BM25 index for local metadata or candidate passages."""

    def __init__(self, texts: list[str]) -> None:
        self.lengths = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        self.average_length = sum(self.lengths) / max(len(texts), 1) or 1.0

    def rank(self, query: str) -> list[tuple[int, float]]:
        """Return positive lexical matches, with deterministic tie breaking."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query, expand_query=True)):
            hits = self.postings.get(term, [])
            idf = math.log(1 + (len(self.lengths) - len(hits) + 0.5) / (len(hits) + 0.5))
            for index, frequency in hits:
                norm = 1.2 * (0.25 + 0.75 * self.lengths[index] / self.average_length)
                scores[index] += idf * frequency * 2.2 / (frequency + norm)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


@dataclass
class Evidence:
    """A verbatim passage whose paragraph range can be checked in the local file."""

    source_id: str
    tdoc: str
    meeting: str
    title: str
    company: str
    file_path: str
    paragraph_start: int
    paragraph_end: int
    text: str

    def context(self) -> str:
        """Format the passage for a model without presenting it as an adopted standard."""
        return (
            f"[{self.source_id}] {self.tdoc} | {self.title}\n"
            f"会议: {self.meeting}; 提交方: {self.company}; "
            f"段落: {self.paragraph_start}-{self.paragraph_end}\n原文: {self.text}"
        )


@dataclass
class RetrievalResult:
    """Keep metadata-only hits separate from readable source evidence."""

    requested_ids: list[str] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        """Require body text for every explicitly requested document."""
        available = {item.tdoc for item in self.evidence}
        return bool(available) and all(tdoc in available for tdoc in self.requested_ids)

    def to_dict(self) -> dict:
        """Serialize for the UI, reports and API responses."""
        return {**asdict(self), "answerable": self.answerable}


class SpecRetrieverTool:
    """Retrieve IDs exactly or rank titles, then select relevant DOCX passages.

    Keyword search covers titles and metadata, not the entire Word corpus.
    Training answers are deliberately excluded from the evidence collection.
    """

    def __init__(
        self,
        proposal_db_path: str | Path | None = None,
        tdoc_cache_path: str | Path = ROOT / "tdoc_index_cache.json",
        docs_root: str | Path = ROOT / "3gpp-tdoc-batch-downloader/RAN2_Rel19_Rel20_Docs",
    ) -> None:
        self.docs_root = Path(docs_root).resolve()
        self.load_warnings: list[str] = []
        self.tdoc_index: dict[str, dict] = {}
        try:
            self.tdoc_index = json.loads(Path(tdoc_cache_path).read_text(encoding="utf-8"))
            if not isinstance(self.tdoc_index, dict):
                raise ValueError("TDoc index must be a JSON object")
        except (OSError, ValueError) as exc:
            self.tdoc_index = {}
            self.load_warnings.append(f"提案索引不可用: {exc}")
        self.ids = sorted(self.tdoc_index)
        self.metadata_index = BM25Index([
            " ".join(str(self.tdoc_index[tdoc].get(key, "")) for key in ("title", "title", "source", "meeting"))
            for tdoc in self.ids
        ])
        self.files: dict[str, list[Path]] = defaultdict(list)
        if self.docs_root.exists():
            for path in sorted(self.docs_root.rglob("*")):
                if path.is_file():
                    for tdoc in extract_tdoc_ids(path.name):
                        self.files[tdoc].append(path)
        else:
            self.load_warnings.append(f"原文目录不存在: {self.docs_root}")
        self._paragraph_cache: dict[Path, tuple[tuple[int, int], list[tuple[int, str]]]] = {}

    def read_paragraphs(self, path: Path) -> list[tuple[int, str]]:
        """Read visible DOCX text, including tables; refresh after file changes."""
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._paragraph_cache.get(path)
        if cached and cached[0] == signature:
            return cached[1]
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > 40_000_000:
                raise ValueError("Word XML exceeds the 40 MB extraction limit")
            tree = ET.fromstring(archive.read(info))
        paragraphs = []
        for number, paragraph in enumerate(tree.iter(WORD_NS + "p"), start=1):
            text = re.sub(r"\s+", " ", "".join(node.text or "" for node in paragraph.iter(WORD_NS + "t"))).strip()
            if text:
                paragraphs.append((number, text))
        self._paragraph_cache[path] = (signature, paragraphs)
        return paragraphs

    def extract_docx_snippet(self, docx_path: str | Path, max_chars: int = 1200) -> str:
        """Compatibility helper; retrieval itself ranks passages across the document."""
        return "\n".join(text for _, text in self.read_paragraphs(Path(docx_path)))[:max_chars]

    @staticmethod
    def _chunks(paragraphs: list[tuple[int, str]], max_chars: int = 900) -> list[tuple[int, int, str]]:
        chunks = []
        current: list[tuple[int, str]] = []
        length = 0
        for number, text in paragraphs:
            for offset in range(0, len(text), max_chars):
                part = text[offset:offset + max_chars]
                if current and length + len(part) + 1 > max_chars:
                    chunks.append((current[0][0], current[-1][0], "\n".join(item[1] for item in current)))
                    current, length = [], 0
                current.append((number, part))
                length += len(part) + 1
        if current:
            chunks.append((current[0][0], current[-1][0], "\n".join(item[1] for item in current)))
        return chunks

    def retrieve_evidence(self, query: str, top_k: int = 4) -> RetrievalResult:
        """Return at most six documents, retaining every requested ID in the status."""
        if not query.strip() or not 1 <= top_k <= 6:
            raise ValueError("请输入问题，检索篇数需在 1 到 6 之间。")
        requested = extract_tdoc_ids(query)
        result = RetrievalResult(requested_ids=requested, warnings=list(self.load_warnings))
        if requested:
            candidates = requested[:6]
            if len(requested) > 6:
                result.warnings.append("一次最多读取 6 篇提案，请缩小问题范围。")
        else:
            candidates = [self.ids[index] for index, _ in self.metadata_index.rank(query)[:top_k]]
        for tdoc in candidates:
            info = self.tdoc_index.get(tdoc)
            record = {"tdoc": tdoc, **(info or {}), "status": "missing", "file_path": ""}
            result.documents.append(record)
            if not info:
                result.warnings.append(f"{tdoc}: 索引中没有此编号，未使用其他提案代替。")
                continue
            paths = self.files.get(tdoc, [])
            matching = [path for path in paths if path.parent.name == info.get("meeting")]
            paths = matching or paths
            docx_paths = [path for path in paths if path.suffix.lower() == ".docx"]
            if not docx_paths:
                record["status"] = "unsupported_format" if paths else "metadata_only"
                record["file_path"] = str(paths[0]) if paths else ""
                result.warnings.append(f"{tdoc}: 只有元数据或未支持的原文格式，不能据此解释提案内容。")
                continue
            # Deterministic selection; do not silently mix revisions of a contribution.
            path = docx_paths[0]
            if not path.resolve().is_relative_to(self.docs_root):
                record["status"] = "read_error"
                result.warnings.append(f"{tdoc}: 文件指向原文目录之外，未读取。")
                continue
            record["file_path"] = str(path)
            if len(docx_paths) > 1:
                result.warnings.append(f"{tdoc}: 有多个 DOCX 文件，本次读取 {path.name}，请核对版本。")
            try:
                chunks = self._chunks(self.read_paragraphs(path))
            except (OSError, ValueError, KeyError, zipfile.BadZipFile, ET.ParseError) as exc:
                record["status"] = "read_error"
                result.warnings.append(f"{tdoc}: 原文读取失败: {exc}")
                continue
            if not chunks:
                record["status"] = "empty_document"
                result.warnings.append(f"{tdoc}: 未提取到可读正文。")
                continue
            record["status"] = "readable"
            ranked = BM25Index([text for _, _, text in chunks]).rank(query)
            scores = dict(ranked)
            # A second passage favors actual proposals/conclusions over repeated headers.
            order = sorted(range(len(chunks)), key=lambda i: (
                -(scores.get(i, 0) + 2 * bool(re.search(r"proposal\s*\d|conclusion|we propose|we suggest", chunks[i][2], re.I))), i
            ))
            for index in order[:2]:
                start, end, text = chunks[index]
                result.evidence.append(Evidence(
                    source_id=f"S{len(result.evidence) + 1}", tdoc=tdoc,
                    meeting=info.get("meeting", ""), title=info.get("title", ""),
                    company=info.get("source", ""), file_path=str(path),
                    paragraph_start=start, paragraph_end=end, text=text,
                ))
        if not candidates:
            result.warnings.append("标题与元数据未找到匹配项，请提供提案编号或英文关键词。")
        return result

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        """Compatibility API returning only readable passages, never training answers."""
        return [item.context() for item in self.retrieve_evidence(query, top_k).evidence]
