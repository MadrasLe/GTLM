"""Training model implementation for GTLM.

This module contains the research architecture used by the training pipeline.
The Hugging Face custom-code files at the repository root are kept separately so
checkpoint loading/inference concerns do not leak into the training code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

    LIGER_RMSNORM_AVAILABLE = True
except ImportError:
    LigerRMSNorm = None
    LIGER_RMSNORM_AVAILABLE = False

try:
    from flash_attn import flash_attn_func

    FLASH_ATTN2_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    FLASH_ATTN2_AVAILABLE = False


class ManualRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def build_norm(hidden_size: int, prefer_liger: bool) -> nn.Module:
    if prefer_liger and LIGER_RMSNORM_AVAILABLE:
        return LigerRMSNorm(hidden_size)
    return ManualRMSNorm(hidden_size)


class ManualSwiGLUExpert(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_multiplier: float,
        hidden_multiple: int,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * hidden_multiplier)
        hidden_dim = hidden_multiple * ((hidden_dim + hidden_multiple - 1) // hidden_multiple)

        self.w_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.activation(self.w_gate(x)) * self.w_up(x))


@dataclass
class MoEForwardOutput:
    hidden_states: torch.Tensor
    load_balance_loss: torch.Tensor
    expert_distribution: torch.Tensor


class OptimizedMoELayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                ManualSwiGLUExpert(
                    config.hidden_dim,
                    config.expert_hidden_multiplier,
                    config.expert_hidden_multiple,
                )
                for _ in range(config.num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> MoEForwardOutput:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)
        num_tokens = x_flat.shape[0]

        logits = self.gate(x_flat).float()
        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)

        probs = F.softmax(logits, dim=-1)
        load_balance_loss = self.num_experts * (probs.mean(0) ** 2).sum()

        flat_topk_indices = topk_indices.reshape(-1)
        token_ids = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)

        sorted_expert_indices, sorted_indices = flat_topk_indices.sort(0)
        permuted_tokens = x_flat[token_ids[sorted_indices]]
        permuted_weights = topk_weights.reshape(-1, 1)[sorted_indices]

        tokens_per_expert = torch.bincount(sorted_expert_indices, minlength=self.num_experts)
        split_tokens = torch.split(permuted_tokens, tokens_per_expert.tolist())

        expert_outputs: list[torch.Tensor] = []
        for expert_id, expert_batch in enumerate(split_tokens):
            if expert_batch.numel() == 0:
                expert_outputs.append(torch.empty(0, hidden_dim, device=x.device, dtype=x.dtype))
                continue
            expert_outputs.append(self.experts[expert_id](expert_batch))

        concatenated_outputs = torch.cat(expert_outputs, dim=0)
        weighted_output = concatenated_outputs * permuted_weights

        output_flat = torch.zeros_like(x_flat)
        output_flat.index_add_(
            0,
            token_ids[sorted_indices],
            weighted_output.to(output_flat.dtype),
        )

        return MoEForwardOutput(
            hidden_states=output_flat.reshape(batch_size, seq_len, hidden_dim),
            load_balance_loss=load_balance_loss,
            expert_distribution=probs.mean(0),
        )


class ManualRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(2).to(x.device).type_as(x)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(2).to(x.device).type_as(x)
        x_reshaped = x.reshape(batch_size, seq_len, num_heads, head_dim // 2, 2)
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]
        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        return torch.stack((y_even, y_odd), dim=-1).reshape(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
        )


class OptimizedAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.dropout_rate = config.dropout_rate
        self.prefer_flash_attention = config.prefer_flash_attention
        self.qkv = nn.Linear(config.hidden_dim, config.hidden_dim * 3, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.rope = ManualRotaryEmbedding(self.head_dim, config.max_seq_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = self.rope(q)
        k = self.rope(k)

        dropout_p = 0.0 if not self.training else self.dropout_rate
        can_use_flash = (
            self.prefer_flash_attention
            and FLASH_ATTN2_AVAILABLE
            and flash_attn_func is not None
            and q.is_cuda
        )

        if can_use_flash:
            attn_out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                causal=True,
                return_attn_probs=False,
            )
            return self.o_proj(attn_out.contiguous().view(batch_size, seq_len, hidden_dim))

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=dropout_p,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.o_proj(attn_out)


class OptimizedTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn = OptimizedAttention(config)
        self.moe = OptimizedMoELayer(config)
        self.ln1 = build_norm(config.hidden_dim, config.prefer_liger_rmsnorm)
        self.ln2 = build_norm(config.hidden_dim, config.prefer_liger_rmsnorm)

    def forward(self, x: torch.Tensor) -> MoEForwardOutput:
        x = x + self.attn(self.ln1(x))
        moe_output = self.moe(self.ln2(x))
        return MoEForwardOutput(
            hidden_states=x + moe_output.hidden_states,
            load_balance_loss=moe_output.load_balance_loss,
            expert_distribution=moe_output.expert_distribution,
        )


class GTLMTrainingModel(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layers = nn.ModuleList(
            [OptimizedTransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.ln_f = build_norm(config.hidden_dim, config.prefer_liger_rmsnorm)
        self.lm_head = nn.Linear(config.hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        total_lb_loss: torch.Tensor | float = 0.0
        expert_metrics = []

        for layer in self.layers:
            layer_output = layer(x)
            x = layer_output.hidden_states
            total_lb_loss = total_lb_loss + layer_output.load_balance_loss
            expert_metrics.append(layer_output.expert_distribution)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        lm_loss = None
        if labels is not None:
            lm_loss = self.criterion(logits.reshape(-1, self.vocab_size), labels.reshape(-1))

        avg_expert_dist = torch.stack(expert_metrics).mean(0)
        avg_lb_loss = total_lb_loss / len(self.layers)
        return lm_loss, avg_lb_loss, avg_expert_dist


def count_parameters(model: GTLMTrainingModel) -> dict[str, int | float]:
    total_params = sum(p.numel() for p in model.parameters())
    params_per_expert = sum(p.numel() for p in model.layers[0].moe.experts[0].parameters())
    total_expert_params = params_per_expert * model.config.num_experts * model.config.num_layers
    shared_params = total_params - total_expert_params
    active_params = shared_params + (
        params_per_expert * model.config.top_k * model.config.num_layers
    )
    utilization_ratio = (active_params / total_params) * 100
    return {
        "total_params": total_params,
        "active_params": active_params,
        "shared_params": shared_params,
        "params_per_expert": params_per_expert,
        "utilization_ratio": utilization_ratio,
    }
