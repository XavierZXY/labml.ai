from typing import List, Optional

import torch
import torch.nn as nn
from labml_helpers.module import Module
from labml_nn.utils import clone_module_list

from ..feed_forward import FeedForward
from .relative_mha import RelativeMultiHeadAttention

# Transformer has a limited attention span, equal to the length of the sequence trained in parallel.
# All these positions have a fixed positional encoding.
# Transformer XL increases this attention span by letting each of the positions pay attention to precalculated past embeddings.
# For instance if the context length is l, it will keep the embeddings of all layers for previous batch of length l
# and feed them to current step. If we use fixed-positional encodings these pre-calculated embeddings will have the
# same positions as the current context. They introduce relative positional encoding, where the positional encodings
# are introduced at the attention calculation.
