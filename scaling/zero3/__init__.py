import functools
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn


class Zero3Layer(nn.Module):
    # Each shard keeps parameters in `chunk` list.
    # The `chunk[0]` is for trainable parameters and `chunk[1]` is for fixed parameters.
    chunk: List[nn.Parameter]
    # This is the sizes of the chunks in `chunk` list.
    chunk_size: List[int]
    # The first chunk is for trainable parameters.
    TRAINING_PARAMS_IDX = 0

    # This is the list of parameters split into lists as trainable and fixed parameters.
    param_refs: List[List[nn.Parameter]]

    # CUDA stream to featch parameters
    fetch_stream: Optional[torch.cuda.Stream]
    # CUDA stream to backup/accumulate gradients
    backup_stream: Optional[torch.cuda.Stream]
    # List of layers right before this layer
    prev_layer: List["Zero3Layer"]
    # List of layers right after this layer
    next_layer: List["Zero3Layer"]
    # The position of the current layer; used this for debugging logs
    layer_idx: int

    # Whether parameters have been fetched
    is_fetched: bool

    # Device of the layer
    device: torch.device
    # Data type of the layer
    dtype: torch.dtype
    # The module to be wrapped
    module: nn.Module
    # Number of nodes/devices the data is sharded across
    world_size: int
