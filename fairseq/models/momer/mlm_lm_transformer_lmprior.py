# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import re
from argparse import Namespace
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerDecoder,
    base_architecture,
)
from fairseq import checkpoint_utils, utils
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerEncoderLayer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from omegaconf import DictConfig
from torch import Tensor


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

from .mlm_encoder import MLMTransformerEncoder
from .my_transfomer_layer import TransformerDecoderLayer

logger = logging.getLogger(__name__)


@register_model("mlm_lm_priorlm_transformer_model")
class MLMLMPriorLMTransformerModel(TransformerModel):
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

    def __init__(self, args, translator, lm):
        super().__init__(args, translator.encoder, translator.decoder)
        if args.freeze_params is not None:
            self.freeze_params(args)

        self.priorlm=lm
        self.priorlm.eval()
        for param in self.priorlm.parameters():
            param.requires_grad = False

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
        parser.add_argument("--mlm", action="store_true", default=False,
                            help="whether train mlm task")
        parser.add_argument("--lm-head-activation-fn", type=str, metavar='D', default='relu',
                            help="mlm head activate")
        parser.add_argument("--encoder-out-layer", type=int, metavar='D', default=-1,
                            help="which encoder out layer for decoding")
        parser.add_argument('--priorlm-checkpoint', metavar='DIR',
                            help='path to load checkpoint from pretrained LM.')

                            
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return MLMTransformerEncoder(
            args,
            src_dict,
            embed_tokens,
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return myTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
    @classmethod
    def build_model(cls, args, task):
        translator = TransformerModel.build_model(args, task)

        # Load checkpoint of pretrained LM
        lm = checkpoint_utils.load_model_ensemble([args.lm_checkpoint])[0][0]

        return cls(args, translator, lm) 


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

    def state_dict(self):
        """
        Omit the parameters of the pretrained LM from the checkpoint
        """
        state = TransformerModel.state_dict(self)
        for k, v in list(state.items()):
            if "priorlm." in k:
                del state[k]
        return state

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Fairseq expects that the returned state_dict should contain weights
        for the pretrained LM as we made it part of the model.
        However, these weights are not stored in the checkpoint and for this
        reason we add them here by copying them from the state_dict of the LM.
        """
        super().upgrade_state_dict_named(state_dict, name)

        # Put the weights of the pretrained LM into the state_dict
        model_state = TransformerModel.state_dict(self)
        for k, v in list(model_state.items()):
            if "priorlm." in k:
                state_dict[k] = v

class myTransformerDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        
    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        only_return_lm_output: bool = False,
    ):
        return super().forward(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            features_only,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            src_lengths,
            return_all_hiddens,
        )
    
    def forward_lm_layers(
        self,
        src_tokens,
        **kwargs,
    ):
        return self.forward(src_tokens, **kwargs)
        
    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"]

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("mlm_lm_priorlm_transformer_model", "mlm_lm_priorlm_transformer")
def base_mlm_lm_transformer(args):
    args.no_cross_attention = getattr(args, "no_cross_attention", True)
    args.cross_self_attention = getattr(args, "cross_self_attention", True)
    base_architecture(args)
    args.encoder_out_layer = getattr(args, "encoder_out_layer", args.encoder_layers)

@register_model_architecture("mlm_lm_priorlm_transformer_model", "mlm_lm_priorlm_transformer_iwslt_de_en")
def mlm_lm_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_mlm_lm_transformer(args)

@register_model_architecture("mlm_lm_priorlm_transformer_model", "mlm_lm_priorlm_transformer_wmt_en_de")
def mlm_lm_transformer_wmt_en_de(args):
    base_mlm_lm_transformer(args)


@register_model_architecture("mlm_lm_priorlm_transformer_model", "mlm_lm_priorlm_transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_mlm_lm_transformer(args)

@register_model_architecture("mlm_lm_priorlm_transformer_model", "mlm_lm_priorlm_transformer_tiny")
def mlm_lm_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 256)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    base_mlm_lm_transformer(args)
