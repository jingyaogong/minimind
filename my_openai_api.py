# encoding: utf-8
import json
import re
import time
import uuid
import warnings

import tiktoken
import torch
import numpy as np
from typing import List
from flask import Flask, current_app, request, Blueprint, stream_with_context
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import PolynomialFeatures
from transformers import AutoTokenizer, AutoModelForCausalLM
from marshmallow import validate, Schema, fields, EXCLUDE
from pydantic import BaseModel

warnings.filterwarnings('ignore', category=UserWarning)

# ------------------------------------------------------------------------------------------------------------------
DEVICE_NAME = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_NAME)
MODEL_PATH = "./minimind-small-T"
TOKENIZE_PATH = MODEL_PATH
max_new_tokens = 1024
temperature = 0.7
top_k = 16


# ------------------------------------------------------------------------------------------------------------------

class Transformers():
    def __init__(self, app=None, tokenizer=None, model=None):
        # self.chat = None
        if app is not None:
            self.init_app(app, tokenizer, model)

    def init_app(self, app, tokenizer=None, model=None, chat=None):
        self.tokenizer = tokenizer
        self.model = model
        # if chat is None:
        #     # self.chat = model.chat
        #     self.chat = self.chat

    # gpt2's
    def build_chat_input(self, tokenizer, messages: List[dict]):
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-(max_new_tokens - 1):]
        inputs_ids = tokenizer(new_prompt).data['input_ids']
        inputs_ids = (torch.tensor(inputs_ids, dtype=torch.long, device=DEVICE)[None, ...])
        return inputs_ids, tokenizer.eos_token_id, new_prompt

    def chat_stream(self, tokenizer, messages: List[dict], stream=True):
        input_ids, eos_token_id, new_prompt = self.build_chat_input(tokenizer, messages)
        if stream:
            res_y = self.model.generate(input_ids, tokenizer.eos_token_id, max_new_tokens=max_new_tokens,
                                        temperature=temperature, top_k=top_k, stream=True)

            try:
                y = next(res_y)
            except:
                print("No answer")
                return 'No answer'

            history_idx = 0
            while y != None:
                answer = tokenizer.decode(y[0].tolist())
                if answer and answer[-1] == '�':
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue
                # print(answer)
                if not len(answer):
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue

                yield answer[history_idx:]
                try:
                    y = next(res_y)
                except:
                    break
                history_idx = len(answer)
                if not stream:
                    break

    def chat_no_stream(self, tokenizer, messages: List[dict]):
        input_ids, eos_token_id, new_prompt = self.build_chat_input(tokenizer, messages)
        res_y = self.model.generate(input_ids, tokenizer.eos_token_id, max_new_tokens=max_new_tokens,
                                    temperature=temperature, top_k=top_k, stream=False)
        y = next(res_y)
        answer = tokenizer.decode(y[0].tolist())
        return answer


tfs = Transformers()
base_tfs = Transformers()

models_bp = Blueprint('Models', __name__, url_prefix='/v1/models')
chat_bp = Blueprint('Chat', __name__, url_prefix='/v1/chat')
completions_bp = Blueprint('Completions', __name__, url_prefix='/v1/completions')
embedding_bp = Blueprint('Embeddings', __name__, url_prefix='/v1')


def sse(line, field="data"):
    return "{}: {}\n\n".format(
        field, json.dumps(line, ensure_ascii=False) if isinstance(line, dict) else line)


def empty_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(models_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(completions_bp)
    app.register_blueprint(embedding_bp)

    @app.after_request
    def after_request(resp):
        empty_cache()
        return resp

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZE_PATH, trust_remote_code=True, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True).to(DEVICE)
    # model.generation_config = GenerationConfig.from_pretrained(model_name)

    tfs.init_app(app, tokenizer, model)
    base_tfs.init_app(app, tokenizer, model)

    return app


class ModelSchema(Schema):
    id = fields.Str()
    object = fields.Str(dump_default="model", metadata={"example": "model"})
    created = fields.Int(dump_default=lambda: int(time.time()), metadata={"example": 1695402567})
    owned_by = fields.Str(dump_default="owner", metadata={"example": "owner"})


class ModelListSchema(Schema):
    object = fields.Str(dump_default="list", metadata={"example": "list"})
    data = fields.List(fields.Nested(ModelSchema), dump_default=[])


class ChatMessageSchema(Schema):
    role = fields.Str(required=True, metadata={"example": "system"})
    content = fields.Str(required=True, metadata={"example": "You are a helpful assistant."})


class CreateChatCompletionSchema(Schema):
    class Meta:
        unknown = EXCLUDE  # 忽略未知的字段

    model = fields.Str(required=True, metadata={"example": "minimind"})
    messages = fields.List(
        fields.Nested(ChatMessageSchema), required=True,
        metadata={"example": [
            ChatMessageSchema().dump({"role": "system", "content": "You are a helpful assistant."}),
            ChatMessageSchema().dump({"role": "user", "content": "Hello!"})
        ]}
    )
    temperature = fields.Float(load_default=1.0, metadata={"example": 1.0})
    top_p = fields.Float(load_default=1.0, metadata={"example": 1.0})
    n = fields.Int(load_default=1, metadata={"example": 1})
    max_tokens = fields.Int(load_default=None, metadata={"example": None})
    stream = fields.Bool(load_default=False, example=False)
    presence_penalty = fields.Float(load_default=0.0, example=0.0)
    frequency_penalty = fields.Float(load_default=0.0, example=0.0)


class ChatCompletionChoiceSchema(Schema):
    index = fields.Int(metadata={"example": 0})
    message = fields.Nested(ChatMessageSchema, metadata={
        "example": ChatMessageSchema().dump(
            {"role": "assistant", "content": "\n\nHello there, how may I assist you today?"}
        )})
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),
        metadata={"example": "stop"})


class ChatCompletionSchema(Schema):
    id = fields.Str(
        dump_default=lambda: uuid.uuid4().hex,
        metadata={"example": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"})
    object = fields.Constant("chat.completion")
    created = fields.Int(dump_default=lambda: int(time.time()), metadata={"example": 1695402567})
    model = fields.Str(metadata={"example": "minimind"})
    choices = fields.List(fields.Nested(ChatCompletionChoiceSchema))


class ChatDeltaSchema(Schema):
    role = fields.Str(metadata={"example": "assistant"})
    content = fields.Str(required=True, metadata={"example": "Hello"})


class ChatCompletionChunkChoiceSchema(Schema):
    index = fields.Int(metadata={"example": 0})
    delta = fields.Nested(ChatDeltaSchema, metadata={"example": ChatDeltaSchema().dump(
        {"role": "assistant", "example": "Hello"})})
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),
        metadata={"example": "stop"})


class ChatCompletionChunkShema(Schema):
    id = fields.Str(
        dump_default=lambda: uuid.uuid4().hex,
        metadata={"example": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"})
    object = fields.Constant("chat.completion.chunk")
    created = fields.Int(dump_default=lambda: int(time.time()), metadata={"example": 1695402567})
    model = fields.Str(metadata={"example": "minimind"})
    choices = fields.List(fields.Nested(ChatCompletionChunkChoiceSchema))


class CreateCompletionSchema(Schema):
    model = fields.Str(required=True, metadata={"example": "minimind"})
    prompt = fields.Raw(metadata={"example": "Say this is a test"})
    max_tokens = fields.Int(load_default=16, metadata={"example": 256})
    temperature = fields.Float(load_default=1.0, metadata={"example": 1.0})
    top_p = fields.Float(load_default=1.0, metadata={"example": 1.0})
    n = fields.Int(load_default=1, metadata={"example": 1})
    stream = fields.Bool(load_default=False, example=False)
    logit_bias = fields.Dict(load_default=None, example={})
    presence_penalty = fields.Float(load_default=0.0, example=0.0)
    frequency_penalty = fields.Float(load_default=0.0, example=0.0)


class CompletionChoiceSchema(Schema):
    index = fields.Int(load_default=0, metadata={"example": 0})
    text = fields.Str(required=True, metadata={"example": "登鹳雀楼->王之涣\n夜雨寄北->"})
    logprobs = fields.Dict(load_default=None, metadata={"example": {}})
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),
        metadata={"example": "stop"})


class CompletionUsageSchema(Schema):
    prompt_tokens = fields.Int(metadata={"example": 5})
    completion_tokens = fields.Int(metadata={"example": 7})
    total_tokens = fields.Int(metadata={"example": 12})


class CompletionSchema(Schema):
    id = fields.Str(
        dump_default=lambda: uuid.uuid4().hex,
        metadata={"example": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"})
    object = fields.Constant("text_completion")
    created = fields.Int(dump_default=lambda: int(time.time()), metadata={"example": 1695402567})
    model = fields.Str(metadata={"example": "minimind"})
    choices = fields.List(fields.Nested(CompletionChoiceSchema))
    usage = fields.Nested(CompletionUsageSchema)


@stream_with_context
def stream_chat_generate(messages):
    delta = ChatDeltaSchema().dump(
        {"role": "assistant"})
    choice = ChatCompletionChunkChoiceSchema().dump(
        {"index": 0, "delta": delta, "finish_reason": None})

    yield sse(
        ChatCompletionChunkShema().dump({
            "model": "minimind",
            "choices": [choice]})
    )

    # 调用 chat 方法并遍历其返回的生成器
    for response in tfs.chat_stream(tfs.tokenizer, messages):
        delta = ChatDeltaSchema().dump(
            {"content": response})
        choice = ChatCompletionChunkChoiceSchema().dump(
            {"index": 0, "delta": delta, "finish_reason": None})

        yield sse(
            ChatCompletionChunkShema().dump({
                "model": "minimind",
                "choices": [choice]})
        )

    yield sse('[DONE]')


@chat_bp.route("/completions", methods=['POST'])
def create_chat_completion():
    create_chat_completion = CreateChatCompletionSchema().load(request.json)

    if create_chat_completion["stream"]:
        return current_app.response_class(
            stream_chat_generate(create_chat_completion["messages"]),
            mimetype="text/event-stream"
        )
    else:
        response = tfs.chat_no_stream(tfs.tokenizer, create_chat_completion["messages"])

        message = ChatMessageSchema().dump(
            {"role": "assistant", "content": response})
        choice = ChatCompletionChoiceSchema().dump(
            {"index": 0, "message": message, "finish_reason": "stop"})

        return ChatCompletionSchema().dump({
            "model": "minimind",
            "choices": [choice]})


class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


@embedding_bp.route("/embeddings", methods=['POST'])
def get_embeddings():
    request_data = request.get_json()  # 获取 POST 请求体中的 JSON 数据
    request_params = EmbeddingRequest(**request_data)  # 将 JSON 数据转换为 EmbeddingRequest 对象

    def expand_features(embedding, target_length):
        poly = PolynomialFeatures(degree=2)
        expanded_embedding = poly.fit_transform(embedding.reshape(1, -1))
        expanded_embedding = expanded_embedding.flatten()
        if len(expanded_embedding) > target_length:
            # 如果扩展后的特征超过目标长度，可以通过截断或其他方法来减少维度
            expanded_embedding = expanded_embedding[:target_length]
        elif len(expanded_embedding) < target_length:
            # 如果扩展后的特征少于目标长度，可以通过填充或其他方法来增加维度
            expanded_embedding = np.pad(
                expanded_embedding, (0, target_length - len(expanded_embedding))
            )
        return expanded_embedding

    def num_tokens_from_string(string: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def has_chinese_char(s):
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        # if bool(pattern.search(s)):
        #     print('m3e编码')
        # else:
        #     print('bge编码')

        return bool(pattern.search(s))

    # 计算嵌入向量和tokens数量
    embeddings = [embeddings_model_m3e.encode(text)
                  if has_chinese_char(text)
                  else embeddings_model_bge.encode(text)
                  for text in request_params.input]

    # 如果嵌入向量的维度不为1536，则使用插值法扩展至1536维度
    embeddings = [
        expand_features(embedding, 768) if len(embedding) < 768 else embedding
        for embedding in embeddings
    ]

    # Min-Max normalization 归一化
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]

    # 将numpy数组转换为列表
    embeddings = [embedding.tolist() for embedding in embeddings]
    prompt_tokens = sum(len(text.split()) for text in request_params.input)
    total_tokens = sum(num_tokens_from_string(text) for text in request_params.input)

    response = {
        "data": [
            {"embedding": embedding, "index": index, "object": "embedding"}
            for index, embedding in enumerate(embeddings)
        ],
        "model": request_params.model,
        "object": "list",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        },
    }
    # print(response)
    return response


app = create_app()

if __name__ == '__main__':
    use_emb = False
    try:
        import ngrok
        import logging

        logging.basicConfig(level=logging.INFO)
        listener = ngrok.werkzeug_develop()
    except Exception:
        pass

    embeddings_model_m3e = SentenceTransformer('.\\m3e-base', device='cpu') if use_emb else None
    embeddings_model_bge = SentenceTransformer('.\\bge-base-en-v1.5', device='cpu') if use_emb else None

    app.run(debug=False, host="0.0.0.0", port=8000)
