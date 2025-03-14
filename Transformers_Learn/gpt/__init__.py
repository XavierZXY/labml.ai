import torch
from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressiveConfigs
from labml_nn.optimizers.configs import OptimizerConfigs
from labml_nn.transformers import Encoder, TransformerConfigs
from labml_nn.transformers.utils import subsequent_mask
from torch import nn


class GPT(Module):
    """GPT model

    Args:
        Module (_type_): _description_
    """

    def __init__(
        self, encoder: Encoder, src_embed: nn.Embedding, generator: Module
    ):
        super().__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.generator = generator
        self.mask = None

    def forward(self, x: torch.Tensor):
        if self.mask is None or self.mask.size(0) != x.size(0):
            self.mask = subsequent_mask(x.size(0)).to(x.device)

        x = self.src_embed(x)
        x = self.encoder(x, self.mask)
        x = self.generator(x)

        return x, None


class Configs(NLPAutoRegressiveConfigs):
    model: GPT
    transformer: TransformerConfigs
    weight_decay: float = 0.1
    warmup_steps: int = 128 * 128 * 20
    optimizer = "transformer_optimizer"


@option(Configs.transformer, "GPT")
def _transformer_configs(c: Configs):
    conf = TransformerConfigs()
    conf.n_src_vocab = c.n_tokens
    conf.n_tgt_vocab = c.n_tokens
    conf.ffn.activation = "gelu"

    return conf


def _init_weight(module):
    if not isinstance(module, nn.Embedding):
        return

    module.weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


@option(Configs.model)
def _model(c: Configs):
    model = GPT(
        c.transformer.encoder,
        c.transformer.src_embed,
        c.transformer.generator.to(c.device),
    )

    model.apply(_init_weight)

    return model


@option(NLPAutoRegressiveConfigs.optimizer)
def transformer_optimizer(c: NLPAutoRegressiveConfigs):
    """Create custom optimizer with weight decay.

    This code is taken from [minGPT](https://github.com/karpathy/minGPT).
    This applies weight decay only to weights of linear layers.
    Args:
        c (NLPAutoRegressiveConfigs): _description_
    """
    decay = set()
    for mn, m in c.model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if fpn.endswith("weight") and isinstance(m, nn.Linear):
                decay.add(fpn)
    param_dict = {pn: p for pn, p in c.model.named_parameters()}

    no_decay = set(param_dict.keys()) - decay

    opt_groups = [
        {
            "params": [param_dict[pn] for pn in decay],
            "weight_decay": c.weight_decay,
        },
        {"params": [param_dict[pn] for pn in no_decay], "weight_decay": 0.0},
    ]

    optimizer = OptimizerConfigs()
    optimizer.parameters = opt_groups
    optimizer.optimizer = "AdamWarmupCosineDecay"
    optimizer.weight_decay = c.weight_decay
    optimizer.learning_rate = 6e-4
    optimizer.betas = (0.9, 0.98)
    optimizer.eps = 1e-9
    optimizer.weight_decouple = True  # 是否将权重衰减与优化器的学习率分开
    optimizer.total_steps = (
        c.epochs * len(c.txtx.train) // (c.batch_size * c.seq_len)
    )
    optimizer.warmup = c.warmup_steps // (c.batch_size * c.seq_len)

    return optimizer


def main():
    # Create experiment
    experiment.create(name="gpt")
    # Create configs
    conf = Configs()
    # Override configurations
    experiment.configs(
        conf,
        {
            # Use character level tokenizer
            "tokenizer": "character",
            # Prompt separator is blank
            "prompt_separator": "",
            # Starting prompt for sampling
            "prompt": "It is ",
            # Use Tiny Shakespeare dataset
            "text": "tiny_shakespeare",
            # Use a context size of $128$
            "seq_len": 128,
            # Train for $32$ epochs
            "epochs": 32,
            # Batch size $128$
            "batch_size": 128,
            # Switch between training and validation for $10$ times
            # per epoch
            "inner_iterations": 10,
            # Transformer configurations
            "transformer.d_model": 512,
            "transformer.ffn.d_ff": 2048,
            "transformer.n_heads": 8,
            "transformer.n_layers": 6,
        },
    )

    # Set models for saving and loading
    experiment.add_pytorch_models({"model": conf.model})

    # Start the experiment
    with experiment.start():
        # Run training
        conf.run()


#
if __name__ == "__main__":
    main()
