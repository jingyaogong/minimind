"""Local HTTP entry points sharing the same lazy inference and evidence workflow."""

import json
import time
import uuid
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_core import ProposalAgent

app = FastAPI(title="3GPP Proposal Assistant")


@lru_cache(maxsize=1)
def get_agent():
    return ProposalAgent()


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=16000)


class ChatCompletionRequest(BaseModel):
    model: str = "local-qwen"
    messages: list[Message] = Field(min_length=1, max_length=32)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    stream: bool = True
    use_lora: bool = False


class AnalysisRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    extra_context: str = Field(default="", max_length=8000)
    retrieval_only: bool = True
    use_lora: bool = False
    max_tokens: int = Field(default=640, ge=1, le=2048)


def encode_sse(data):
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(get_agent().runtime.model is not None)}


@app.post("/v1/proposals/analyze")
def analyze(req: AnalysisRequest):
    """SSE workflow including sources, explicit failures and retrieval-only operation."""
    def events():
        for event in get_agent().run_stream(
            req.question, extra_context=req.extra_context, retrieval_only=req.retrieval_only,
            use_lora=req.use_lora, max_new_tokens=req.max_tokens,
        ):
            yield encode_sse(event)
        yield "data: [DONE]\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """Basic chat-compatible endpoint; raw chat does not run source retrieval."""
    completion_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())

    def response_base(kind):
        return {"id": completion_id, "object": kind, "created": created, "model": req.model}

    chunks = get_agent().runtime.stream(
        [message.model_dump() for message in req.messages], req.max_tokens, req.temperature, req.use_lora,
    )
    if not req.stream:
        try:
            text = "".join(chunks)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {**response_base("chat.completion"), "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ]}

    def events():
        try:
            yield encode_sse({**response_base("chat.completion.chunk"), "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ]})
            for chunk in chunks:
                yield encode_sse({**response_base("chat.completion.chunk"), "choices": [
                    {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
                ]})
            yield encode_sse({**response_base("chat.completion.chunk"), "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ]})
        except Exception as exc:
            yield encode_sse({"error": {"message": str(exc), "type": type(exc).__name__}})
        finally:
            chunks.close()
        yield "data: [DONE]\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
