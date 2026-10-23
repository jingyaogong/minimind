"""Regression tests for evidence retrieval, refusals, citations and streaming failures."""

import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

from agent_core import LocalModelRuntime, ProposalAgent, SpecRetrieverTool, SpecValidatorTool
from rag_evidence import ROOT, Evidence, WORD_NS, extract_tdoc_ids


class FixtureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "output")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = self.root / "docs"
        self.meeting = self.docs / "meeting"
        self.meeting.mkdir(parents=True)
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps({
            "R2-2400001": {"meeting": "meeting", "title": "XR rate control", "source": "A"},
            "R2-2400002": {"meeting": "meeting", "title": "Delay-aware LCP", "source": "B"},
            "R2-2400003": {"meeting": "meeting", "title": "Mobility", "source": "C"},
            "R2-2400004": {"meeting": "meeting", "title": "Metadata only", "source": "D"},
        }), encoding="utf-8")
        self.write_docx("R2-2400001", ["Header " + "x" * 1800, "Proposal 1: recommended bit rate signaling over a DRB."])
        self.write_docx("R2-2400002", ["Delay-aware LCP uses remaining packet delay information."])
        (self.meeting / "R2-2400003.doc").write_bytes(b"unsupported legacy document")
        self.retriever = SpecRetrieverTool(tdoc_cache_path=self.index, docs_root=self.docs)

    def write_docx(self, number, paragraphs):
        document = ET.Element(WORD_NS + "document")
        body = ET.SubElement(document, WORD_NS + "body")
        for text in paragraphs:
            paragraph = ET.SubElement(body, WORD_NS + "p")
            run = ET.SubElement(paragraph, WORD_NS + "r")
            ET.SubElement(run, WORD_NS + "t").text = text
        with zipfile.ZipFile(self.meeting / (number + ".docx"), "w") as archive:
            archive.writestr("word/document.xml", ET.tostring(document))

    def test_multiple_ids_survive_top_k_one(self):
        result = self.retriever.retrieve_evidence("compare R2-2400001 and R2-2400002", top_k=1)
        self.assertTrue(result.answerable)
        self.assertEqual({item.tdoc for item in result.evidence}, {"R2-2400001", "R2-2400002"})

    def test_proposal_after_long_header_is_retrieved(self):
        result = self.retriever.retrieve_evidence("R2-2400001 recommended bit rate")
        passage = next(item for item in result.evidence if "Proposal 1" in item.text)
        self.assertLessEqual(passage.paragraph_start, 2)
        self.assertGreaterEqual(passage.paragraph_end, 2)
        self.assertLessEqual(len(passage.text), 900)

    def test_unknown_id_does_not_substitute_training_answer(self):
        result = self.retriever.retrieve_evidence("R2-2999999 XR rate control")
        self.assertFalse(result.answerable)
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.documents[0]["status"], "missing")

    def test_partial_comparison_is_not_answerable(self):
        result = self.retriever.retrieve_evidence("R2-2400001 R2-2999999")
        self.assertTrue(result.evidence)
        self.assertFalse(result.answerable)

    def test_legacy_doc_is_not_body_evidence(self):
        result = self.retriever.retrieve_evidence("R2-2400003")
        self.assertFalse(result.answerable)
        self.assertEqual(result.documents[0]["status"], "unsupported_format")

    def test_metadata_only_is_not_body_evidence(self):
        result = self.retriever.retrieve_evidence("R2-2400004")
        self.assertFalse(result.answerable)
        self.assertEqual(result.documents[0]["status"], "metadata_only")

    def test_bm25_search_finds_title(self):
        result = self.retriever.retrieve_evidence("速率控制", top_k=1)
        self.assertEqual(result.documents[0]["tdoc"], "R2-2400001")

    def test_no_lexical_match_returns_no_evidence(self):
        self.assertEqual(self.retriever.retrieve_evidence("zzzxqmissing").evidence, [])

    def test_corrupt_docx_is_reported(self):
        (self.meeting / "R2-2400002.docx").write_bytes(b"invalid zip")
        result = self.retriever.retrieve_evidence("R2-2400002")
        self.assertFalse(result.answerable)
        self.assertEqual(result.documents[0]["status"], "read_error")

    def test_changed_docx_refreshes_paragraph_cache(self):
        self.retriever.retrieve_evidence("R2-2400002")
        self.write_docx("R2-2400002", ["A newly revised paragraph about logical channels."])
        result = self.retriever.retrieve_evidence("R2-2400002")
        self.assertIn("newly revised", result.evidence[0].text)

    def test_missing_index_has_visible_warning(self):
        retriever = SpecRetrieverTool(tdoc_cache_path=self.root / "absent.json", docs_root=self.docs)
        self.assertTrue(retriever.retrieve_evidence("XR").warnings)

    def test_abstention_never_calls_runtime(self):
        class MustNotRun:
            def stream(self, *args):
                raise AssertionError("No model should run without evidence")
        agent = ProposalAgent(retriever=self.retriever, runtime=MustNotRun())
        last = list(agent.run_stream("R2-2400003"))[-1]
        self.assertEqual(last["mode"], "insufficient_evidence")

    def test_disabled_retrieval_respected_even_with_id(self):
        agent = ProposalAgent(retriever=self.retriever)
        last = list(agent.run_stream("R2-2400001", enable_rag=False))[-1]
        self.assertEqual(last["sources"]["evidence"], [])
        self.assertIsNone(agent.runtime.model)

    def test_retrieval_only_never_loads_model(self):
        agent = ProposalAgent(retriever=self.retriever)
        last = list(agent.run_stream("R2-2400001", retrieval_only=True))[-1]
        self.assertEqual(last["mode"], "retrieval_only")
        self.assertIsNone(agent.runtime.model)

    def test_model_receives_both_sources_and_marks_extra_unverified(self):
        class RecordingRuntime:
            messages = None
            def stream(self, messages, *args):
                self.messages = messages
                yield "速率控制见 [S1]。时延处理见 [S3]。"
        runtime = RecordingRuntime()
        agent = ProposalAgent(retriever=self.retriever, runtime=runtime)
        last = list(agent.run_stream("R2-2400001 R2-2400002", extra_context="custom claim"))[-1]
        self.assertEqual(last["mode"], "generated")
        self.assertIn("R2-2400002", runtime.messages[1]["content"])
        self.assertIn("未经核实", runtime.messages[1]["content"])
        self.assertIn("first_text_seconds", last["metrics"])

    def test_generation_error_is_not_reported_as_completion(self):
        class BrokenRuntime:
            def stream(self, *args):
                yield "partial answer"
                raise RuntimeError("GPU unavailable")
        agent = ProposalAgent(retriever=self.retriever, runtime=BrokenRuntime())
        events = list(agent.run_stream("R2-2400001"))
        self.assertEqual(events[-1]["stage"], "ERROR")
        self.assertFalse(any(event["stage"] == "4. COMPLETE" for event in events))

    def test_empty_question_is_an_error(self):
        self.assertEqual(list(ProposalAgent(retriever=self.retriever).run_stream(" "))[-1]["stage"], "ERROR")


class CitationTests(unittest.TestCase):
    def test_normalize_and_deduplicate_ids(self):
        self.assertEqual(extract_tdoc_ids("r2 – 2408133 R2-2408133 R2-2408134"), ["R2-2408133", "R2-2408134"])

    def test_invented_citation_fails(self):
        evidence = [Evidence("S1", "R2-2400001", "m", "title", "A", "file", 1, 2, "source")]
        result = SpecValidatorTool().validate("A statement [S99]", evidence)
        self.assertEqual(result["unknown_citations"], ["S99"])
        self.assertEqual(result["factual_correctness"], "not_verified")
        self.assertNotIn("compliance_score", result)

    def test_known_citation_does_not_claim_factually_verified(self):
        evidence = [Evidence("S1", "R2-2400001", "m", "title", "A", "file", 1, 2, "source")]
        result = SpecValidatorTool().validate("A wrong statement [S1]", evidence)
        self.assertEqual(result["citation_status"], "format_checked")
        self.assertEqual(result["factual_correctness"], "not_verified")

    def test_comparison_citing_only_one_document_fails(self):
        evidence = [
            Evidence("S1", "R2-2400001", "m", "title", "A", "file", 1, 2, "source"),
            Evidence("S2", "R2-2400002", "m", "title", "B", "file", 1, 2, "source"),
        ]
        result = SpecValidatorTool().validate("Both proposals do this [S1]", evidence)
        self.assertEqual(result["citation_status"], "failed")
        self.assertEqual(result["uncited_documents"], ["R2-2400002"])

    def test_no_citations_is_not_a_fact_check(self):
        result = SpecValidatorTool().validate("TS 38.331 RRC")
        self.assertEqual(result["factual_correctness"], "not_verified")
        self.assertNotIn("is_valid", result)


class RuntimeTests(unittest.TestCase):
    @staticmethod
    def runtime(behavior):
        import torch
        from transformers import BatchEncoding

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 2
            def apply_chat_template(self, *args, **kwargs):
                return "prompt"
            def __call__(self, *args, **kwargs):
                return BatchEncoding({"input_ids": torch.tensor([[1, 3]]), "attention_mask": torch.tensor([[1, 1]])})
            def decode(self, tokens, **kwargs):
                return "".join("word " for _ in tokens)

        class Model:
            config = SimpleNamespace(max_position_embeddings=8192)
            def get_input_embeddings(self):
                return SimpleNamespace(weight=torch.zeros(1))
            def generate(self, **kwargs):
                kwargs["streamer"].put(kwargs["input_ids"])
                if behavior == "error":
                    raise RuntimeError("intentional worker failure")
                if behavior == "timeout":
                    time.sleep(0.15)
                else:
                    kwargs["streamer"].put(torch.tensor([4]))
                kwargs["streamer"].end()

        runtime = LocalModelRuntime()
        runtime.model, runtime.tokenizer = Model(), Tokenizer()
        return runtime

    def test_worker_error_reaches_caller_and_releases_lock(self):
        runtime = self.runtime("error")
        with self.assertRaisesRegex(RuntimeError, "intentional worker failure"):
            list(runtime.stream([{"role": "user", "content": "q"}]))
        self.assertFalse(runtime._lock.locked())

    def test_stream_deadline_and_lock_cleanup(self):
        runtime = self.runtime("timeout")
        with self.assertRaises(TimeoutError):
            list(runtime.stream([{"role": "user", "content": "q"}], timeout_seconds=0.03))
        self.assertFalse(runtime._lock.locked())

    def test_successful_stream_returns_text(self):
        runtime = self.runtime("success")
        self.assertIn("word", "".join(runtime.stream([{"role": "user", "content": "q"}])))

    def test_busy_request_is_rejected(self):
        runtime = self.runtime("success")
        runtime._lock.acquire()
        self.addCleanup(runtime._lock.release)
        with self.assertRaisesRegex(RuntimeError, "另一条请求"):
            list(runtime.stream([{"role": "user", "content": "q"}]))


if __name__ == "__main__":
    unittest.main()
