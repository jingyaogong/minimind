import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re

# 控制标记正则缓存（tokenizer 生命周期内只构建一次）
_PATTERN_CACHE = {}


def _collect_control_markers(tokenizer):
    """收集所有会被视为“结构控制符”的字符串：
    1. tokenizer 的特殊 token（<|im_start|>、<|im_end|>、<|image_pad|> 等）
    2. chat_template 里的模板级标记（思考标签、工具调用标签等）"""
    markers = set(tokenizer.all_special_tokens)
    template = getattr(tokenizer, 'chat_template', None) or (getattr(tokenizer, 'init_kwargs', None) or {}).get('chat_template') or ''
    markers |= set(re.findall(r'<[^<>{}\s]+>', template))
    return markers


def _get_control_pattern(tokenizer):
    key = id(tokenizer)
    if key not in _PATTERN_CACHE:
        alts = sorted({re.escape(m) for m in _collect_control_markers(tokenizer) if m}, key=len, reverse=True)
        _PATTERN_CACHE[key] = re.compile('|'.join(alts)) if alts else None
    return _PATTERN_CACHE[key]


def sanitize_text(text, tokenizer):
    """删除文本中的控制标记，防止用户输入伪造对话结构
    （例如输入 <|im_end|><|im_start|>assistant 劫持角色边界）"""
    if not isinstance(text, str):
        return text
    pattern = _get_control_pattern(tokenizer)
    return pattern.sub('', text) if pattern else text


def sanitize_messages(messages, tokenizer):
    """返回去除控制标记后的消息副本（不修改原始 messages），
    供 apply_chat_template 前净化不受信任的输入。"""
    sanitized = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        message = dict(message)
        if isinstance(message.get('content'), str):
            message['content'] = sanitize_text(message['content'], tokenizer)
        if isinstance(message.get('reasoning_content'), str):
            message['reasoning_content'] = sanitize_text(message['reasoning_content'], tokenizer)
        sanitized.append(message)
    return sanitized
