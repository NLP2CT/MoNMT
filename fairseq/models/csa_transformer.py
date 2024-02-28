# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq.models.transformer import (
    TransformerModel,
    TransformerDecoder,
    base_architecture,
    register_model,
)
from fairseq.modules import LayerNorm
from fairseq.models import register_model_architecture
from fairseq import utils
from omegaconf import DictConfig
from torch import Tensor


logger = logging.getLogger(__name__)


@register_model("csa_transformer_model")
class csaTransformerModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        if args.freeze_params is not None:
            self.freeze_params(args)

    def freeze_params(self, args):
        freeze_pattern = re.compile(args.freeze_params)
        for name, parameter in self.named_parameters():
            if freeze_pattern.search(name):
                parameter.requires_grad = False
                logger.info(f"Freeze: {name}")
        for name, parameter in self.named_parameters():
            if not freeze_pattern.search(name):
                logger.info(f"Unfreeze: {name}")

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument('--freeze-params', type=str, metavar='D', default=None,
                            help='regular expression of parameters that need to be frozen')
        parser.add_argument('--transfer-params', type=str, metavar='D', default=None,
                            help='transfer params from pretrained models')

    def load_state_dict(
        self,
        state_dict,
        strict=False,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        if self.args.transfer_params is not None and "inference" not in vars(model_cfg):
            pretrained_model_prefix = [*state_dict][0].split('.')[0]
            pairs = self.args.transfer_params.split(',')
            for pair in pairs:
                from_param, to_param = pair.split(':')
                if from_param in state_dict:
                    # state_dict[to_param] = state_dict.pop(from_param)
                    state_dict[to_param] = state_dict[from_param]
                    logger.info(f"Transfer {from_param} to {to_param} in model [{pretrained_model_prefix}]")
        return torch.nn.Module.load_state_dict(self, state_dict, strict=False)


@register_model_architecture("csa_transformer_model", "csa_transformer")
def bridge_transformer_model(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_architecture(args)
