# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import Dict, List, Optional

import torch
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
    FairseqEncoderDecoderModel
)
# improt torch.nn as nn

logger = logging.getLogger(__name__)


DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("hf_mT5_MoNMT")
class HuggingFacemT5forMoNMT(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.reencoder=None

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--embed-dim', type=int, metavar='N',
                            help='embedding dimension')
        parser.add_argument('--num-attention-heads', type=int, metavar='N',
                            help='num attention heads')
        parser.add_argument('--num-layers', type=int, metavar='N',
                            help='num layers')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability for all fully connected layers '
                                 'in the embeddings, encoder, and pooler')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        default_architecture(args)
        mT5 = HuggingFacemT5(args)
        return cls(mT5.encoder, mT5.decoder)


class HuggingFacemT5(nn.Module):
    def __init__(self, args):
        try:
            from transformers import MT5Config, MT5Model
        except ImportError:
            raise ImportError(
                "\n\nPlease install huggingface/transformers with:"
                "\n\n  pip install transformers"
            )

        super().__init__()

        # config = GPT2Config(
        #     vocab_size=len(task.target_dictionary),
        #     n_positions=args.max_target_positions + 1,
        #     n_ctx=args.max_target_positions,
        #     n_embd=args.embed_dim,
        #     n_layer=args.num_layers,
        #     n_head=args.num_attention_heads,
        #     resid_pdrop=args.dropout,
        #     embd_pdrop=args.dropout,
        #     attn_pdrop=args.attention_dropout,
        #     layer_norm_epsilon=1e-6,
        # )
        self.model = MT5Model.from_pretrained("mt5-small")


@register_model_architecture("hf_gpt2", "hf_gpt2")
def default_architecture(args):
    if getattr(args, "max_target_positions", None) is None:
        args.max_target_positions = getattr(
            args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
        )
    args.embed_dim = getattr(args, "embed_dim", 768)
    args.num_attention_heads = getattr(args, "num_attention_heads", 12)
    args.num_layers = getattr(args, "num_layers", 12)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)


@register_model_architecture("hf_gpt2", "hf_gpt2_medium")
def hf_gpt2_medium(args):
    args.embed_dim = getattr(args, "embed_dim", 1024)
    args.num_attention_heads = getattr(args, "num_attention_heads", 16)
    args.num_layers = getattr(args, "num_layers", 24)
    default_architecture(args)


@register_model_architecture("hf_gpt2", "hf_gpt2_large")
def hf_gpt2_large(args):
    args.embed_dim = getattr(args, "embed_dim", 1280)
    args.num_attention_heads = getattr(args, "num_attention_heads", 20)
    args.num_layers = getattr(args, "num_layers", 36)
    default_architecture(args)


@register_model_architecture("hf_gpt2", "hf_gpt2_xl")
def hf_gpt2_xl(args):
    args.embed_dim = getattr(args, "embed_dim", 1600)
    args.num_attention_heads = getattr(args, "num_attention_heads", 25)
    args.num_layers = getattr(args, "num_layers", 48)
    default_architecture(args)
