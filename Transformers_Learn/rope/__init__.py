"""
Author: XavierZXY
Date: 2025-02-24 19:07:54
LastEditors: XavierZXY
Description: Rotary Position Embedding

Copyright (c) 2025 by XavierZXY, All Rights Reserved.
"""

import torch
from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base: int = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        if self.cos_cached is None and x.shape <= self.cos_cached.shape[0]:
            return
        seq_len = x.shape[0]
        # $theta_i = 10_000^{-{\frac_{d}^{2(i-1)}}}$
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.d, 2).float / self.d)
        ).to(x.device)
        # create the position indexes [0, 1, ..., seq_len - 1]
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        idx_theta = torch.einsum("n, d -> nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        # cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.d // 2
        return torch.cat(-x[:, :, :, d_2:], x[:, :, :, :d_2], dim=-1)

    def forward(self, x: torch.Tensor):
        self._build_cache(x)
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[: x.shape[0]]) + (
            neg_half_x * self.sin_cached[: x.shape[0]]
        )

        return torch.cat((x_rope, x_pass), dim=-1)


class RotaryPEMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        heads: int,
        d_model: int,
        rope_percentage: float = 0.5,
        dropout_prob: float = 0.0,
    ):
        super().__init__(heads, d_model, dropout_prob)
        d_rope = int(self.d_k * rope_percentage)
        self.query_rotary_pe = RotaryPositionalEmbeddings(d_rope)
        self.key_rotary_pe = RotaryPositionalEmbeddings(d_rope)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum(
            "ibhd,jbhd->ijbh",
            self.query_rotary_pe(query),
            self.key_rotary_pe(key),
        )


def _test_rotary():
    x = torch.tensor(
        [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=torch.float
    )
    x = x[:, None, None, :]
    inspect(x)

    rotary_pe = RotaryPositionalEmbeddings(4)
    inspect(rotary_pe(x))


if __name__ == "__main__":
    _test_rotary()
