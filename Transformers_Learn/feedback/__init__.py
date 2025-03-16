import math
from typing import Optional

import torch
from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.utils import clone_module_list
from torch import nn


class FeedbackAttention(Module):
    def __init__(
        self,
        heads: int,
        d_model: int,
        dropout_prob: float = 0.1,
        *args,
        is_kv_precomputed: bool = False,
    ):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(
            d_model, heads, self.d_k, bias=False
        )
        if not is_kv_precomputed:
            self.key = PrepareForMultiHeadAttention(
                d_model, heads, self.d_k, bias=False
            )
            self.value = PrepareForMultiHeadAttention(
                d_model, heads, self.d_k, bias=False
            )
        else:
            self.key = None
            self.value = None

        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.softmax = nn.Softmax(dim=0)
        self.P = 2**12
        self.key_pos_embeddings = nn.Parameter(
            torch.zeros((self.P, self.d_k)), requires_grad=True
        )
        self.key_pos_bias = nn.Parameter(
            torch.zeros((self.P, self.heads)), requires_grad=True
        )
        self.query_pos_bias = nn.Parameter(
            torch.zeros((heads, self.d_k)), requires_grad=True
        )
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        key_pos_emb = self.key_pos_embeddings[-key.shape[0] :]
        query_pos_bias = self.query_pos_bias[None, :, :]
        key_pos_bias = self.key_pos_bias[-key.shape[0] :]

    # TODO: Implement the rest of the function
