"""
MiniMind with Multi-head Latent Attention (MLA)
基于 DeepSeek-V2 的低秩 KV 压缩注意力机制（解耦 RoPE 架构）
将 GQA 中独立的 K/V 投影替换为"压缩→缓存→还原"的低秩路径
位置信息通过独立的低维投影承载（decoupled rope），K 的 content 部分不施 RoPE
KV-Cache 从 GQA 的 768 floats/token/layer 压缩到 kv_lora_rank + rope_dim floats（默认 128+48=176，压缩约 4.4 倍）
"""
import math, torch, torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from model.model_minimind import (
    MiniMindConfig, RMSNorm, precompute_freqs_cis, repeat_kv,
    FeedForward, MOEFeedForward
)


class MiniMindMLAConfig(MiniMindConfig):
    model_type = "minimind_mla"

    def __init__(self, kv_lora_rank=128, q_lora_rank=256, rope_dim=None, **kwargs):
        kwargs["attention_type"] = "mla"
        super().__init__(**kwargs)
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        # hidden_size=768 → head_dim=96 → rope_dim=48
        self.rope_dim = rope_dim if rope_dim is not None else self.hidden_size // self.num_attention_heads // 2


def _apply_rope(x, cos, sin, unsqueeze_dim=1):
    """单张量 RoPE，MLA 中 Q 和 K 序列长度不同，需要分别应用"""
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    return ((x * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(x) * sin.unsqueeze(unsqueeze_dim))).to(x.dtype)


class MLAAttention(nn.Module):
    def __init__(self, config: MiniMindMLAConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.rope_dim = config.rope_dim
        self.is_causal = True

        # Q 低秩压缩路径 (DeepSeek-V2 Eq.12-14)
        self.q_compress = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_up = nn.Linear(config.q_lora_rank, config.num_attention_heads * self.head_dim, bias=False)
        self.q_rope_proj = nn.Linear(config.q_lora_rank, config.num_attention_heads * self.rope_dim, bias=False)

        # KV 低秩压缩路径 (DeepSeek-V2 Eq.9-11)
        self.kv_compress = nn.Linear(config.hidden_size, config.kv_lora_rank, bias=False)
        self.kv_up_k = nn.Linear(config.kv_lora_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.kv_up_v = nn.Linear(config.kv_lora_rank, self.num_key_value_heads * self.head_dim, bias=False)

        # K 的解耦 RoPE 投影 (DeepSeek-V2 Eq.15)
        self.k_rope_proj = nn.Linear(config.hidden_size, config.rope_dim, bias=False)

        # 输出投影
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # 归一化
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.kv_norm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        cos, sin = position_embeddings  # [seq_len, rope_dim]

        # ── Q Path ──
        q_latent = self.q_compress(x)  # [bsz, seq, q_lora_rank]
        # Q content: q_compress → q_up → q_norm (NO RoPE)
        q_content = self.q_up(q_latent).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        q_content = self.q_norm(q_content)  # [bsz, seq, 8, 96]
        # Q rope: q_compress → q_rope_proj → RoPE
        q_rope = self.q_rope_proj(q_latent).view(bsz, seq_len, self.n_local_heads, self.rope_dim)
        q_rope = _apply_rope(q_rope, cos, sin)  # [bsz, seq, 8, 48]
        # 拼接 content + rope
        xq = torch.cat([q_content, q_rope], dim=-1)  # [bsz, seq, 8, 144]

        # ── K Path ──
        # Content: kv_compress → norm → cache → kv_up_k (NO RoPE!)
        kv_latent = self.kv_norm(self.kv_compress(x))  # [bsz, seq, kv_lora_rank]
        # Rope: k_rope_proj → RoPE (共享跨所有 head)
        k_rope = self.k_rope_proj(x).unsqueeze(2)  # [bsz, seq, 1, rope_dim]
        k_rope = _apply_rope(k_rope, cos, sin).squeeze(2)  # [bsz, seq, rope_dim]

        # 拼接缓存
        if past_key_value is not None:
            past_kv_latent, past_k_rope = past_key_value
            kv_latent = torch.cat([past_kv_latent, kv_latent], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=1)
        past_kv = (kv_latent, k_rope) if use_cache else None

        total_len = kv_latent.shape[1]
        # 从 latent 还原 K content (NO RoPE!)
        k_content = self.kv_up_k(kv_latent).view(bsz, total_len, self.n_local_kv_heads, self.head_dim)  # [bsz, total, 4, 96]
        # K rope 扩展 head 维度后拼接
        k_rope_expanded = k_rope.unsqueeze(2).expand(-1, -1, self.n_local_kv_heads, -1)  # [bsz, total, 4, rope_dim]
        xk = torch.cat([k_content, k_rope_expanded], dim=-1)  # [bsz, total, 4, 144]

        # ── V Path ──
        xv = self.kv_up_v(kv_latent).view(bsz, total_len, self.n_local_kv_heads, self.head_dim)  # [bsz, total, 4, 96]

        # ── Attention ──
        scale = math.sqrt(self.head_dim + self.rope_dim)
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / scale
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class MiniMindMLABlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindMLAConfig):
        super().__init__()
        self.self_attn = MLAAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindMLAModel(nn.Module):
    def __init__(self, config: MiniMindMLAConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindMLABlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.rope_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # MLA 的 cache 是 tuple (kv_latent, k_rope)，取 kv_latent 的长度
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.rope_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        # MLA: RoPE 只作用于当前 token 的 rope 投影（Q rope 和 K rope 使用相同位置范围）
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


class MiniMindMLAForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindMLAConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMindMLAConfig = None):
        self.config = config or MiniMindMLAConfig()
        super().__init__(self.config)
        self.model = MiniMindMLAModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            # MLA: past_key_values[0] 是 tuple (kv_latent, k_rope)，取 kv_latent 的长度
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
