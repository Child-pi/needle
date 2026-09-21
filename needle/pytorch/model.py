import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NeedleConfig:
    vocab_size: int = 16384
    d_model: int = 768
    num_heads: int = 12
    num_kv_heads: int = 2
    num_layers: int = 20
    qk_head_dim: int = 48
    v_head_dim: int = 64
    max_seq_len: int = 4096
    pad_token_id: int = 0
    embedding_dim: int = 128
    confidence_probes: int = 4
    confidence_queries: int = 4
    rope_theta: float = 100000.0
    engram_orders: Tuple[int, ...] = (2, 3)
    engram_heads: int = 0
    engram_slots: int = 18432
    engram_layers: Tuple[int, ...] = (3, 7, 11, 15, 19)
    global_layers: Tuple[int, ...] = (4, 9, 14, 19)
    sliding_window: int = 1024
    mhc_lanes: int = 4
    qkv_conv_taps: int = 3
    out_vocab: int = 0
    torch_dtype: str = "float32"

    def head_dims(self) -> Tuple[int, int]:
        legacy = self.d_model // self.num_heads
        qk = self.qk_head_dim or legacy
        v = self.v_head_dim or legacy
        return qk, v


class ZCRMSNorm(nn.Module):
    """Zero-Centered RMSNorm matching Flax ZCRMSNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Initialized to zeros, so (1 + scale) starts as 1.0
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS along last dimension
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        normed = x.float() / rms
        return (normed * (1.0 + self.scale)).to(x.dtype)


def precompute_rope_freqs(head_dim: int, seq_len: int, theta: float = 100000.0, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, num_heads, T, head_dim]
    T = x.shape[2]
    half = x.shape[-1] // 2
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, half]
    sin = sin[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, half]
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated.to(x.dtype)


def _shift_right(x: torch.Tensor, offset: int) -> torch.Tensor:
    if offset == 0:
        return x
    # shift along sequence dimension (dim 1)
    zeros = torch.zeros(x.shape[0], offset, *x.shape[2:], dtype=x.dtype, device=x.device)
    shifted = torch.cat([zeros, x[:, :-offset]], dim=1)
    return shifted


class HadamardMLP(nn.Module):
    """Monarch Hadamard MLP in place of standard dense FFN."""
    def __init__(self, d_model: int, rank: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n = 1 << (d_model - 1).bit_length()
        ba = 1 << ((self.n - 1).bit_length() // 2)
        bb = self.n // ba

        # Normalized Walsh-Hadamard matrices
        def walsh_matrix(size):
            H = torch.tensor([[1.0]], dtype=torch.float32)
            while H.shape[0] < size:
                H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
            return H / math.sqrt(size)

        self.w1a = nn.Parameter(walsh_matrix(ba))
        self.w1b = nn.Parameter(walsh_matrix(bb))
        self.w2a = nn.Parameter(walsh_matrix(ba))
        self.w2b = nn.Parameter(walsh_matrix(bb))
        self.w3a = nn.Parameter(walsh_matrix(ba))
        self.w3b = nn.Parameter(walsh_matrix(bb))

        self.d1 = nn.Parameter(torch.ones(self.n))
        self.d2 = nn.Parameter(torch.ones(self.n))
        self.b2 = nn.Parameter(torch.zeros(self.n))
        self.d3 = nn.Parameter(torch.ones(self.n))
        self.d4 = nn.Parameter(torch.full((self.n,), 0.02))

        self.cond_v = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.cond_u = nn.Parameter(torch.zeros(rank, self.n))

        # Permutations
        rng1 = torch.Generator().manual_seed(11)
        rng2 = torch.Generator().manual_seed(13)
        self.register_buffer("p1", torch.randperm(self.n, generator=rng1))
        self.register_buffer("p2", torch.randperm(self.n, generator=rng2))

    def _kron_apply(self, z: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        lead = z.shape[:-1]
        z = z.view(*lead, a.shape[0], b.shape[0])
        # einsum: "...ij,ik,jl->...kl"
        z = torch.einsum("...ij,ik,jl->...kl", z, a, b)
        return z.reshape(*lead, a.shape[0] * b.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        cond = 1.0 + torch.softmax(x @ self.cond_v, dim=-1) @ self.cond_u
        pad = self.n - self.d_model
        z = F.pad(x, (0, pad)) if pad else x

        z = self._kron_apply(self.d1 * z, self.w1a, self.w1b)[..., self.p1]
        z = self._kron_apply(F.silu(self.d2 * cond * z + self.b2), self.w2a, self.w2b)[..., self.p2]
        z = self._kron_apply(self.d3 * z, self.w3a, self.w3b)
        out = (self.d4 * z)[..., :self.d_model]
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config: NeedleConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.d_model = config.d_model
        qk_hd, v_hd = config.head_dims()
        self.qk_head_dim = qk_hd
        self.v_head_dim = v_hd
        self.out_dim = config.num_heads * v_hd
        self.qkv_conv_taps = config.qkv_conv_taps

        self.q_proj = nn.Linear(config.d_model, config.num_heads * qk_hd, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * qk_hd, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * v_hd, bias=False)
        self.gate_proj = nn.Linear(config.d_model, self.out_dim, bias=False)
        self.out_proj = nn.Linear(self.out_dim, config.d_model, bias=False)

        self.q_norm = ZCRMSNorm(qk_hd)
        self.k_norm = ZCRMSNorm(qk_hd)

        if self.qkv_conv_taps:
            n_taps = self.qkv_conv_taps
            self.q_taps = nn.Parameter(torch.zeros(n_taps, config.num_heads * qk_hd))
            self.k_taps = nn.Parameter(torch.zeros(n_taps, config.num_kv_heads * qk_hd))
            self.v_taps = nn.Parameter(torch.zeros(n_taps, config.num_kv_heads * v_hd))
            # Identity initialization: tap 0 = 1.0
            with torch.no_grad():
                self.q_taps[0] = 1.0
                self.k_taps[0] = 1.0
                self.v_taps[0] = 1.0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.qkv_conv_taps:
            q_sum = torch.zeros_like(q)
            k_sum = torch.zeros_like(k)
            v_sum = torch.zeros_like(v)
            for j in range(self.qkv_conv_taps):
                q_sum = q_sum + self.q_taps[j] * _shift_right(q, j)
                k_sum = k_sum + self.k_taps[j] * _shift_right(k, j)
                v_sum = v_sum + self.v_taps[j] * _shift_right(v, j)
            q, k, v = q_sum, k_sum, v_sum

        # Reshape to [B, num_heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.qk_head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.qk_head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.v_head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # GQA expansion
        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        scale = 1.0 / math.sqrt(self.qk_head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, num_heads, T, v_head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, self.out_dim)

        gate = torch.sigmoid(self.gate_proj(x))
        out = out * gate
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: NeedleConfig):
        super().__init__()
        self.d_model = config.d_model
        self.self_attn = MultiHeadAttention(config)
        self.pre_attn_norm = ZCRMSNorm(config.d_model)
        self.post_attn_norm = ZCRMSNorm(config.d_model)
        self.pre_hada_norm = ZCRMSNorm(config.d_model)
        self.hadamard_mlp = HadamardMLP(config.d_model)
        self.attn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                engram_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                site_flag: float = 0.0) -> torch.Tensor:
        # Engram attention fusion if present
        if engram_kv is not None and site_flag > 0.0:
            ek, ev = engram_kv
            def rms_u(t):
                return t / torch.sqrt(torch.mean(t.float() ** 2, dim=-1, keepdim=True) + 1e-6)
            alpha = torch.sigmoid(torch.einsum("btd,sbtd->sbt", rms_u(x), rms_u(ek)) / math.sqrt(self.d_model))
            x = x + site_flag * torch.einsum("sbt,sbtd->btd", alpha, ev.to(x.dtype))

        # Attention block
        skip = x
        normed = self.pre_attn_norm(x)
        attn_out = self.self_attn(normed, mask=mask, rope=rope)
        attn_out = self.post_attn_norm(attn_out)
        x = skip + torch.sigmoid(self.attn_gate) * attn_out

        # Hadamard MLP block
        skip = x
        normed = self.pre_hada_norm(x)
        mlp_out = self.hadamard_mlp(normed)
        return skip + mlp_out


class NeedleModel(nn.Module):
    def __init__(self, config: NeedleConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_scale = math.sqrt(config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = ZCRMSNorm(config.d_model)

        # Confidence Head
        self.confidence_proj = nn.Linear(config.d_model, 1)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        device = input_ids.device

        # Causal mask
        if mask is None:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)

        # RoPE freqs
        qk_hd, _ = self.config.head_dims()
        rope = precompute_rope_freqs(qk_hd, T, self.config.rope_theta, device=device)

        x = self.embedding(input_ids) * self.embed_scale

        for layer in self.layers:
            x = layer(x, mask=mask, rope=rope)

        x = self.final_norm(x)

        # Tied LM Head
        weight = self.embedding.weight
        if self.config.out_vocab:
            weight = weight[:self.config.out_vocab]
        logits = F.linear(x, weight)

        # Calibrated Confidence Score
        pooled = torch.mean(x, dim=1)
        confidence = torch.sigmoid(self.confidence_proj(pooled)).squeeze(-1)

        return logits, confidence


class NeedleForCausalLM(nn.Module):
    """High-level PyTorch wrapper matching Hugging Face / standard LLM API."""
    def __init__(self, config: Optional[NeedleConfig] = None):
        super().__init__()
        self.config = config or NeedleConfig()
        self.model = NeedleModel(self.config)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None):
        logits, confidence = self.model(input_ids, mask=mask)
        return {"logits": logits, "confidence": confidence}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 0.0, eos_token_id: int = 2) -> torch.Tensor:
        curr = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self.model(curr)
            next_logits = out[0][:, -1, :]
            if temperature <= 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            curr = torch.cat([curr, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break
        return curr
