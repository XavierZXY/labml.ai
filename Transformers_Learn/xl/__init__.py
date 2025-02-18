from typing import List, Optional

import torch
import torch.nn as nn
from labml_helpers.module import Module
from labml_nn.utils import clone_module_list

from ..feed_forward import FeedForward
from .relative_mha import RelativeMultiHeadAttention

# https://zhuanlan.zhihu.com/p/271984518
# Transformer has a limited attention span, equal to the length of the sequence trained in parallel.
# All these positions have a fixed positional encoding.
# Transformer XL increases this attention span by letting each of the positions pay attention to precalculated past embeddings.
# For instance if the context length is l, it will keep the embeddings of all layers for previous batch of length l
# and feed them to current step. If we use fixed-positional encodings these pre-calculated embeddings will have the
# same positions as the current context. They introduce relative positional encoding, where the positional encodings
# are introduced at the attention calculation.


class TransformerXLLayer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        self_attn: RelativeMultiHeadAttention,
        feed_forward: FeedForward,
        dropout_prob: float,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def forward(
        self,
        *,
        x: torch.Tensor,  # [seq_len, batch_size, d_model]
        mem: Optional[
            torch.Tensor
        ],  # a tensor of the pask token level feature, [mem_len, batch_size, d_model]
        mask: torch.Tensor,
    ):
        # noramlize the vectors before doing self attention
        z = self.norm_self_attn(x)
        if mem is not None:
            mem = self.norm_self_attn(mem)
            m_z = torch.cat([mem, z], dim=0)
        else:
            m_z = z
        self_attn = self.self_attn(z, m_z, m_z, mask)
        x = x + self.dropout(self_attn)
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        return x


class TransformerXL(Module):
    def __init__(self, layer: TransformerXLLayer, num_layers: int):
        super().__init__()
        self.layers = clone_module_list(layer, num_layers)
        self.norm = nn.LayerNorm([layer.size])

    def forward(
        self, x: torch.Tensor, mem: List[torch.Tensor], mask: torch.Tensor
    ):
        new_mem = []
        for i, layer in enumerate(self.layers):
            new_mem.append(x.detach())
            m = mem[i] if mem else None
            x = layer(x=x, mem=m, mask=mask)
        return self.norm(x), new_mem
