"""
Author: XavierZXY
Date: 2025-03-03 18:57:34
LastEditors: XavierZXY
Description:
    This is an implementation of Attention with Linear Biases (ALiBi) from the paper Train Short,
    Test Long: Attention with Linear Biases Enables Input Length Extrapolation.
    https://nn.labml.ai/transformers/alibi/index.html
Copyright (c) 2025 by XavierZXY, All Rights Reserved.
"""

import math
from typing import Optional

import torch
from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention
from torch import nn


def get_slopes(n_heads: int):
    """get head-specific slopes m for each head

    Args:
        n_heads (int): _description_
    """
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)  # 2^{\frac{-8}{n}}
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-2.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n)), 2)
        m = torch.cat([m, m_hat])
    return m


@torch.no_grad()
def get_alibi_biases(n_heads: int, mask: torch.Tensor):
    m = get_slopes(n_heads).to(mask.device)
    distance = mask.cumsum(dim=-1)

    return distance[:, :, None] * m[None, None, :]


class AlibiMultiHeadAttention(MultiHeadAttention):
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.0):
        super().__init__(n_heads, d_model, dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        assert mask is not None, "Mask is required for ALiBi"
        assert mask.shape[0] == mask.shape[1] and mask.shape[2] == 1, (
            "Mask should be square and 1D"
        )
        seq_len, batch_size, _ = query.shape
        mask = self.prepare_mask(mask, query.shape, key.shape)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        scores = self.get_scores(query, key)
        scores *= self.scale

        if self.alibi_biases is not None or self.alibi_biases.shape[1] < seq_len:
            self.alibi_biases = get_alibi_biases(
                scores.shape[-1], mask[:, :, 0, 0]
            )

        scores += self.alibi_biases[:seq_len, :seq_len, None, :]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.softmax(scores)

        attn = self.dropout(attn)
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)
        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x)


def _test_alibi():
    inspect(get_slopes(12).tolist(), _n=-1)
    from labml_nn.transformers.utils import subsequent_mask

    mask = subsequent_mask(8)[:, :, 0]
    inspect(mask)
    inspect(get_alibi_biases(12, mask)[:, :, 3], _n=-1)


if __name__ == "__main__":
    _test_alibi()
