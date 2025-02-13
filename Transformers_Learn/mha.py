import math
from typing import List, Optional

import torch
from labml import tracker
from torch import nn


class PrepareForMultiHeadAttention(nn.Module):
    """
    This moudule does a linear transformation and splits the vector into given number of heads.
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        """_summary_

        Args:
            d_model (int): input dimension
            heads (int): number of heads
            d_k (int): the dimension of the key, query and value vectors
            bias (bool): _description_
        """
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # x: [seq_len, batch_size, d_model]
        head_shape = x.shape[:-1]
        # x = self.linear(x)  # [seq_len, batch_size, heads * d_k]
        x = x.view(*head_shape, self.heads, self.d_k)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        heads: int,
        d_model: int,
        dropout_prob: float = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.value = PrepareForMultiHeadAttention(
            d_model, heads, self.d_k, bias=True
        )

        # softmax
        self.softmax = nn.Softmax(dim=-1)

        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum("ibhd,jbhd->ijbh", query, key)

    def prepare_mask(
        self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]
    ):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)
        return mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key)
        scores = scores * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(scores)
        tracker.debug("attn", attn)
        attn = self.dropout(attn)
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        self.attn = attn.detach()

        x = x.reshape(seq_len, batch_size, -1)
        # return self.output(x)
        return x


def test_multi_head_attention():
    # Set random seeds for reproducibility
    torch.manual_seed(42)

    d_model = 512
    heads = 8
    seq_len = 10
    batch_size = 16

    query = torch.randn(seq_len, batch_size, d_model)
    key = torch.randn(seq_len, batch_size, d_model)
    value = torch.randn(seq_len, batch_size, d_model)

    # My implementation
    multi_head_attention = MultiHeadAttention(heads, d_model)
    output = multi_head_attention(query, key, value)

    # PyTorch implementation
    torch_mha = nn.MultiheadAttention(d_model, heads)
    torch_output, _ = torch_mha(
        query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
    )
    torch_output = torch_output.transpose(0, 1)

    assert output.shape == torch_output.shape
    # Compare the outputs
    torch.testing.assert_close(output, torch_output, rtol=1e-05, atol=1e-05)


if __name__ == "__main__":
    test_multi_head_attention()
