"""
Author: XavierZXY
Date: 2025-02-18 16:27:22
LastEditors: XavierZXY
Description: Relative Multi-Headed Attention

Copyright (c) 2025 by XavierZXY, All Rights Reserved.
"""

import torch
from labml.logger import inspect
from torch import nn

from ..mha import MultiHeadedAttention


def shift_right(x: torch.Tensor):
    """[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    ->  [
        [1, 2, 3],
        [0, 4, 5],
        [6, 0, 7]
    ]

    Args:
        x (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    print(x.shape)
    x_padded = torch.cat([x, zero_pad], dim=1)
    inspect(x_padded)
    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    inspect(x_padded)
    x = x_padded[:-1].view_as(x)
    return x


class RelativeMultiHeadAttention(MultiHeadedAttention):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__(heads, d_model, dropout_prob, bias=False)
        self.P = 2**12
        self.key_pos_embeddings = nn.Parameter(
            torch.zeros((self.P * 2, heads, self.d_k)), requires_grad=True
        )
        self.key_pos_bias = nn.Parameter(
            torch.zeros((self.P * 2, heads)), requires_grad=True
        )
        # Positional embeddings for the query is independent of the position of the query
        self.query_pos_bias = nn.Parameter(
            torch.zeros((heads, self.d_k)), requires_grad=True
        )

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """Get relative attention scores

        Args:
            query (torch.Tensor): _description_
            key (torch.Tensor): _description_
        """
        # $\textcolor{orange}{R_k}$
        key_pos_emb = self.key_pos_embeddings[
            self.P - key.shape[0] : self.P + query.shape[0]
        ]
        # $\textcolor{orange}{S_k}$
        key_pos_bias = self.key_pos_bias[
            self.P - key.shape[0] : self.P + query.shape[0]
        ]
        # $\textcolor{orange}{v^\top}$
        query_pos_bias = self.query_pos_bias[None, None, :, :]

        # ${(\textcolor{lightgreen}{\mathbf{A + C}})}_{i,j} =
        # Q_i^\top K_j +
        # \textcolor{orange}{v^\top} K_j$
        ac = torch.einsum("ibhd,jbhd->ijbh", query + query_pos_bias, key)
        # $\textcolor{lightgreen}{\mathbf{B'}_{i,k}} = Q_i^\top \textcolor{orange}{R_k}$
        b = torch.einsum("ibhd,jhd->ijbh", query, key_pos_emb)
        # $\textcolor{lightgreen}{\mathbf{D'}_{i,k}} = \textcolor{orange}{S_k}$
        d = key_pos_bias[None, :, None, :]
        # Shift the rows of $\textcolor{lightgreen}{\mathbf{(B' + D')}_{i,k}}$
        # to get $$\textcolor{lightgreen}{\mathbf{(B + D)}_{i,j} = \mathbf{(B' + D')}_{i,i - j}}$$
        bd = shift_right(b + d)
        # Remove extra positions
        bd = bd[:, -key.shape[0] :]

        # Return the sum $$
        # \underset{\mathbf{\textcolor{lightgreen}{A}}}{Q_i^\top K_j} +
        # \underset{\mathbf{\textcolor{lightgreen}{B}}}{Q_i^\top \textcolor{orange}{R_{i - j}}} +
        # \underset{\mathbf{\textcolor{lightgreen}{C}}}{\textcolor{orange}{v^\top} K_j} +
        # \underset{\mathbf{\textcolor{lightgreen}{D}}}{\textcolor{orange}{S_{i-j}}}
        # $$
        return ac + bd


def _test_shift_right():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    inspect(x)
    inspect(shift_right(x))

    x = torch.arange(1, 6)[None, :, None, None].repeat(5, 1, 1, 1)
    inspect(x[:, :, 0, 0])
    inspect(shift_right(x)[:, :, 0, 0])

    x = torch.arange(1, 6)[None, :, None, None].repeat(3, 1, 1, 1)
    inspect(x[:, :, 0, 0])
    inspect(shift_right(x)[:, :, 0, 0])


if __name__ == "__main__":
    inspect(shift_right(x=torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
