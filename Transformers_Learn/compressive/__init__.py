from typing import List, Optional

import torch
import torch.nn.functional as F
from labml_helpers.module import Module, TypedModuleList
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.transformers.xl.relative_mha import RelativeMultiHeadAttention
from labml_nn.utils import clone_module_list
from torch import nn


class Conv1dCompression(Module):
    """一维卷积压缩

    Args:
        Module (_type_): _description_
    """

    def __init__(self, compression_rate: int, d_model: int):
        """_summary_

        Args:
            compression_rate (int): compression rate
            d_model (int): the number of features in the input
        """
        super().__init__()
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=compression_rate,
            stride=compression_rate,
        )

    def forward(self, mem: torch.Tensor):
        """_summary_

        Args:
            mem (torch.Tensor): [seq_len, batch_size, d_model]
        """
        mem = mem.permute(1, 2, 0)
        c_mem = self.conv(mem)
        return c_mem.permute(2, 0, 1)


class CompressiveTransformerLayer(Module):
    ### 压缩变换器层
    def __init__(
        self,
        d_model: int,
        self_attn: RelativeMultiHeadAttention,
        feed_forward: FeedForward,
        drouput_prob: float,
        compress: Conv1dCompression,
    ):
        super().__init__()
        self.compress = compress
        self.size = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(drouput_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def concat_memory(
        self,
        z: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
        c_mem: Optional[torch.Tensor] = None,
    ):
        """将标准化令牌嵌入与内存和压缩内存连接起来

        Args:
            z (torch.Tensor): 层归一化后的令牌嵌入
            mem (Optional[torch.Tensor], optional): 内存
            c_mem (Optional[torch.Tensor], optional): 压缩后的内存
        """
        if mem is None:
            return z
        if c_mem is not None:
            mem = torch.cat([mem, c_mem], dim=0)
        mem = self.norm_self_attn(mem)
        return torch.cat([mem, z], dim=0)

    def forward(
        self,
        x: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
        c_mem: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
    ):
        """_summary_

        Args:
            x (torch.Tensor): [seq_len, batch_size, d_model]
            mem (Optional[torch.Tensor], optional): [mem_len, batch_size, d_model]
            c_mem (Optional[torch.Tensor], optional): [c_mem_len, batch_size, d_model]
            mask (torch.Tensor): [seq_len, seq_len]
        """
        z = self.norm_self_attn(x)
        m_z = self.concat_memory(z, mem, c_mem)
        self_attn = self.self_attn(query=z, key=m_z, value=m_z, mask=mask)
        x = x + self.dropout(self_attn)
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        return x


class CompressiveTransformer(Module):
    def __init__(
        self,
        layer: CompressiveTransformerLayer,
        n_layers: int,
    ):
        super().__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])

    def forward(
        self,
        x: torch.Tensor,
        mem: List[torch.Tensor],
        c_mem: List[torch.Tensor],
        mask: torch.Tensor,
    ):
        """_summary_

        Args:
            x (torch.Tensor): _description_
            mem (List[torch.Tensor]): _description_
            c_mem (List[torch.Tensor]): _description_
            mask (torch.Tensor): _description_
        """
        new_mem = []  # 用于存储令牌级特征向量的列表，这些向量将成为下一个连续批次的记忆
        for i, layer in enumerate(self.layers):
            new_mem.append(x.detach())
            m = mem[i] if mem is not None else None
            c_m = c_mem[i] if c_mem is not None else None
            x = layer(x, m, c_m, mask)

        return self.norm(x), new_mem


class AttentionReconstructionLoss:
    """注意力重建损失使用未压缩的内存和压缩的内存重建注意力输出，并计算两者之间的方差"""

    def __init__(
        self,
        layers: TypedModuleList[CompressiveTransformerLayer],
    ):
        self.layers = layers
        self.loss_func = nn.MSELoss()

    def prepare_for_attn(
        self, pmha: PrepareForMultiHeadAttention, x: torch.Tensor
    ):
        """_summary_

        Args:
            pmha (PrepareForMultiHeadAttention): _description_
            x (torch.Tensor): _description_
        """
        head_shape = x.shape[:-1]  # [seq_len, batch_size]
        # 分离投影权重和偏差
        weight = pmha.linear.weight.detach()
        bias = (
            pmha.linear.bias.detach() if pmha.linear.bias is not None else None
        )
        # 投影
        x = F.linear(x, weight, bias)
        x = x.view(
            *head_shape, pmha.heads, pmha.d_k
        )  # [seq_len, batch_size, heads, d_k]

        return x

    def attn(
        self,
        layer: RelativeMultiHeadAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """多头注意力的重新实现

        Args:
            layer (RelativeMultiHeadAttention): _description_
            query (torch.Tensor): _description_
            key (torch.Tensor): _description_
            value (torch.Tensor): _description_
        """
        query = self.prepare_for_attn(layer.query, query)
        key = self.prepare_for_attn(layer.key, key)
        value = self.prepare_for_attn(layer.value, value)

        scores = torch.einsum("ibhd,jbhf->ijbh", query, key)
        scores *= layer.scale
        attn = layer.softmax(scores)

        return torch.einsum("ijbh,jbhf->ibhd", attn, value)

    def norm(self, ln: nn.LayerNorm, x: torch.Tensor):
        weight = ln.weight.detach() if ln.weight is not None else None
        bias = ln.bias.detach() if ln.bias is not None else None

        return F.layer_norm(x, ln.normalized_shape, weight, bias, ln.eps)

    def cacl_loss(
        self,
        layer: CompressiveTransformerLayer,
        h: torch.Tensor,
        mem: torch.Tensor,
    ):
        h = h.detach()
        mem = mem.detach()
        c_mem = layer.compress(mem)

        h = self.norm(layer.norm_self_attn, h)
        mem = self.norm(layer.norm_self_attn, mem)
        c_mem = self.norm(layer.norm_self_attn, c_mem)
        atten_mem = self.attn(layer.self_attn, h, mem, mem)
        atten_c_mem = self.attn(layer.self_attn, h, c_mem, c_mem)

        return self.loss_func(atten_mem, atten_c_mem)

    def __call__(
        self,
        h: List[torch.Tensor],
        mem: List[torch.Tensor],
    ):
        losses = [
            self.cacl_loss(layer, h[n], mem[n])
            for n, layer in enumerate(self.layers)
        ]

        return sum(losses)
