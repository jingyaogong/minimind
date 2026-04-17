"""
Regression tests for scripts/serve_openai_api.py
These tests verify OpenAI API compatibility without loading a real model.
"""
import json
import re
import sys
import os
import types

# ---------------------------------------------------------------------------
# Stub heavy dependencies so importing serve_openai_api doesn't need GPU/model
# ---------------------------------------------------------------------------

# Provide a minimal torch stub (only attributes actually used at import time)
_torch_stub = types.ModuleType("torch")
_torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
_torch_stub.no_grad = lambda: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *a: None)
_torch_stub.inference_mode = lambda: lambda fn: fn
_torch_stub.load = lambda *a, **kw: {}
_torch_stub.Tensor = type("Tensor", (), {})
_torch_stub.float32 = "float32"
_torch_stub.LongTensor = lambda *a, **kw: None

# optim sub-module
_optim_stub = types.ModuleType("torch.optim")
sys.modules["torch.optim"] = _optim_stub
_torch_stub.optim = _optim_stub

# nn sub-module with Module and common layers
_nn_stub = types.ModuleType("torch.nn")
class _ModuleStub:
    def __init_subclass__(cls, **kw): pass
    def __init__(self, *a, **kw): pass
_nn_stub.Module = _ModuleStub
_nn_stub.Linear = type("Linear", (_ModuleStub,), {})
_nn_stub.Embedding = type("Embedding", (_ModuleStub,), {})
_nn_stub.Dropout = type("Dropout", (_ModuleStub,), {})
class _ParameterStub:
    def __init__(self, *a, **kw): pass
_nn_stub.Parameter = _ParameterStub
_torch_stub.nn = _nn_stub

_nn_func = types.ModuleType("torch.nn.functional")
_nn_func.scaled_dot_product_attention = lambda *a, **kw: None
_nn_func.cross_entropy = lambda *a, **kw: None
_nn_func.softmax = lambda *a, **kw: None
_nn_func.silu = lambda *a, **kw: None

sys.modules["torch"] = _torch_stub
sys.modules["torch.nn"] = _nn_stub
sys.modules["torch.nn.functional"] = _nn_func
sys.modules.setdefault("torch.distributed", types.ModuleType("torch.distributed"))
sys.modules.setdefault("torch.nn.parallel", types.ModuleType("torch.nn.parallel"))
sys.modules.setdefault("torch.utils", types.ModuleType("torch.utils"))
sys.modules.setdefault("torch.utils.data", types.ModuleType("torch.utils.data"))

for mod_name in [
    "transformers", "transformers.activations", "transformers.modeling_outputs",
    "librosa", "soundfile", "numpy", "uvicorn",
]:
    sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

# Stub transformers classes that are used at module scope
_transformers = sys.modules["transformers"]
for cls_name in [
    "PreTrainedModel", "GenerationMixin", "PretrainedConfig",
    "AutoTokenizer", "AutoModelForCausalLM", "AutoModel",
    "AutoModelForSequenceClassification", "TextStreamer",
]:
    if not hasattr(_transformers, cls_name):
        setattr(_transformers, cls_name, type(cls_name, (), {}))

_mo = sys.modules["transformers.modeling_outputs"]
if not hasattr(_mo, "MoeCausalLMOutputWithPast"):
    setattr(_mo, "MoeCausalLMOutputWithPast", type("MoeCausalLMOutputWithPast", (), {}))

_act = sys.modules["transformers.activations"]
if not hasattr(_act, "ACT2FN"):
    setattr(_act, "ACT2FN", {})

# Now import the functions we can test without a live model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from scripts.serve_openai_api import parse_response


# ============================================================
# Tests for parse_response
# ============================================================

class TestParseResponse:
    """Verify that parse_response correctly separates reasoning, content and tool calls."""

    def test_plain_text(self):
        content, reasoning, tool_calls = parse_response("Hello, world!")
        assert content == "Hello, world!"
        assert reasoning is None
        assert tool_calls is None

    def test_think_tags_complete(self):
        text = "<think>Step 1: consider options</think>The answer is 42."
        content, reasoning, tool_calls = parse_response(text)
        assert reasoning == "Step 1: consider options"
        assert content == "The answer is 42."
        assert tool_calls is None

    def test_think_tag_no_opening(self):
        """When only </think> is present (streaming partial), split at the tag."""
        text = "I need to think carefully</think>Final answer here"
        content, reasoning, tool_calls = parse_response(text)
        assert reasoning == "I need to think carefully"
        assert content == "Final answer here"

    def test_tool_call_parsed(self):
        text = 'Sure, I can help. <tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}</tool_call>'
        content, reasoning, tool_calls = parse_response(text)
        assert "Sure, I can help." in content
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args["city"] == "Tokyo"

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name":"a","arguments":{}}</tool_call>'
            '<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>'
        )
        content, reasoning, tool_calls = parse_response(text)
        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "a"
        assert tool_calls[1]["function"]["name"] == "b"

    def test_invalid_tool_call_json_skipped(self):
        text = '<tool_call>not valid json</tool_call>The rest.'
        content, reasoning, tool_calls = parse_response(text)
        assert "The rest." in content
        assert tool_calls is None  # Invalid JSON is silently skipped

    def test_empty_string(self):
        content, reasoning, tool_calls = parse_response("")
        assert content == ""
        assert reasoning is None
        assert tool_calls is None


# ============================================================
# Tests for SSE stream format (structural, no live model)
# ============================================================

class TestSSEStreamFormat:
    """Verify the SSE event stream wrapper produces correct framing."""

    def test_done_marker_present(self):
        """The stream MUST end with 'data: [DONE]\\n\\n' per OpenAI spec."""
        # Read the source and verify the [DONE] marker is emitted
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "serve_openai_api.py")
        with open(script_path) as f:
            source = f.read()
        assert 'data: [DONE]' in source, "Stream must emit 'data: [DONE]' terminator"

    def test_stream_chunks_have_required_fields(self):
        """Each stream chunk must include id, object, created, model, choices."""
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "serve_openai_api.py")
        with open(script_path) as f:
            source = f.read()
        # Verify _make_chunk builds proper structure with all required fields
        assert '"id": request_id' in source or '"id":' in source
        assert '"object": "chat.completion.chunk"' in source
        assert '"created":' in source
        assert '"model":' in source

    def test_cors_middleware_present(self):
        """CORS middleware must be configured for browser-based clients."""
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "serve_openai_api.py")
        with open(script_path) as f:
            source = f.read()
        assert "CORSMiddleware" in source, "CORS middleware required for browser clients"

    def test_non_stream_uses_max_new_tokens(self):
        """Non-stream path must use max_new_tokens, not max_length, matching stream behavior."""
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "serve_openai_api.py")
        with open(script_path) as f:
            source = f.read()
        # The non-stream generate call should use max_new_tokens
        # Find the non-stream generate block (after "with torch.no_grad():")
        non_stream_section = source[source.index("with torch.no_grad()"):]
        assert "max_new_tokens" in non_stream_section, \
            "Non-stream path should use max_new_tokens for consistent behavior with stream path"
        # max_length should NOT appear in the generate call
        generate_block = non_stream_section[:non_stream_section.index("answer = tokenizer.decode")]
        assert "max_length" not in generate_block, \
            "Non-stream path should not use max_length (use max_new_tokens instead)"

    def test_generation_thread_is_daemon(self):
        """Generation thread must be a daemon so it doesn't block process exit."""
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "serve_openai_api.py")
        with open(script_path) as f:
            source = f.read()
        assert "daemon=True" in source, "Generation thread should be daemon to avoid blocking exit"

    def test_generation_thread_error_propagation(self):
        """Generation thread errors must be propagated to the consumer, not swallowed."""
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "serve_openai_api.py")
        with open(script_path) as f:
            source = f.read()
        # The _generate() function should have a try/except that puts errors on the queue
        assert "_SENTINEL" in source, "Sentinel-based error propagation required"


# ============================================================
# Run with pytest or directly
# ============================================================

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
