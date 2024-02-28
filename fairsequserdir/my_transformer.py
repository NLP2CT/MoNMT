import torch
import argparse
import logging
import re
from typing import Optional, Dict, List, NamedTuple
import torch.nn as nn
import fairseq.checkpoint_utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerDecoder, TransformerEncoder, TransformerModel, base_architecture, transformer_vaswani_wmt_en_fr_big
from fairseq.models.roberta import model as roberta
from fairseq import checkpoint_utils
from collections import OrderedDict
from .multilingual_reencoder import ReEncoder
from torch import Tensor
from fairseq.models.momer import MLMTransformerEncoder, CLMTransformerDecoder

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

logger = logging.getLogger(__name__)


@register_model("my_transformer")
class myTransformer(TransformerModel):

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        
        parser.add_argument('--freeze-params', type=str, metavar='D', default=None,
                            help='regular expression of parameters that need to be frozen')
        parser.add_argument('--transfer-params', type=str, metavar='D', default=None,
                            help='transfer params from pretrained models')

        parser.add_argument('--reencoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--reencoder-ffn-embed-dim', type=int, metavar='N', help='encoder embedding dimension for FFN')
        parser.add_argument('--reencoder-attention-heads', type=int, metavar='N', help='num encoder attention heads')
        parser.add_argument('--reencoder-normalize-before', action='store_true', help='apply layernorm before each encoder block')

        parser.add_argument('--reencoder-layerdrop', type=float, metavar='D', default=0, help='LayerDrop probability for reencoder')

        parser.add_argument('--reencoder-layers-to-keep', default=None, help='which layers to *keep* when pruning as a comma-separated list')

        parser.add_argument(
            "--reencoder-layers",
            type=int,
            metavar="D",
            default=6,
            help="the number of reencoder layers"
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            default=None,
            type=str,
            metavar="PRETRAINED",
            help="path to pretrained mlm checkpoint",
        )

        parser.add_argument(
            "--load-pretrained-decoder-from",
            default=None,
            type=str,
            metavar="PRETRAINED",
            help="path to pretrained mlm checkpoint",
        )

        parser.add_argument(
            "--load-pretrained-reencoder-from",
            default=None,
            type=str,
            metavar="PRETRAINED",
            help="path to pretrained mlm checkpoint",
        )
        

        # arguments for mlm and clm
        parser.add_argument("--mlm", action="store_true", default=False,
                            help="whether train mlm task")
        parser.add_argument("--lm-head-activation-fn", type=str, metavar='D', default='relu',
                            help="mlm head activate")
        parser.add_argument("--encoder-out-layer", type=int, metavar='D', default=-1,
                            help="which encoder out layer for decoding")

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

    def load_state_dict(
        self,
        state_dict,
        strict=False,
        model_cfg = None,
        args = None,
    ):
        if self.args.transfer_params is not None and "inference" not in vars(model_cfg):
            pretrained_model_prefix = [*state_dict][0].split('.')[0]
            pairs = self.args.transfer_params.split(',')
            for pair in pairs:
                from_param, to_param = pair.split(':')
                if from_param in state_dict:
                    state_dict[to_param] = state_dict.pop(from_param)
                    # state_dict[to_param] = state_dict[from_param]
                    logger.info(f"Transfer {from_param} to {to_param} in model [{pretrained_model_prefix}]")
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, False, model_cfg, args)
        # missing_keys, unexpected_keys =  torch.nn.Module.load_state_dict(self, state_dict, strict=False)
        logger.info(f"pretrained model missing kyes | : {missing_keys}")
        logger.info(f"pretrained model unexpect kyes | : {unexpected_keys}")
        return missing_keys, unexpected_keys

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            logger.info(f"load pretrained model for encoder")
            model_state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(args.load_pretrained_encoder_from, arg_overrides=None)["model"]
            keys=list(model_state.keys())
            for k in keys:
                if k.startswith('decoder.'):
                    del model_state[k]
                if k.startswith('encoder.'):
                    model_state[k[8:]] = model_state.pop(k)
            missing_keys, unexpect_keys = encoder.load_state_dict(model_state, strict=False)
            logger.info(f"pretrained encoder missing kyes | : {missing_keys}")
            logger.info(f"pretrained encoder unexpect kyes | : {unexpect_keys}")
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

        if getattr(args, "load_pretrained_decoder_from", None):
            logger.info(f"load pretrained model for decoder")
            model_state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(args.load_pretrained_decoder_from, arg_overrides=None)["model"]
            keys=list(model_state.keys())
            for k in keys:
                if k.startswith('encoder.'):
                    del model_state[k]
                if k.startswith('decoder.'):
                    model_state[k[8:]] = model_state.pop(k)
            missing_keys, unexpect_keys = decoder.load_state_dict(model_state, strict=False)
            logger.info(f"pretrained decoder missing kyes | : {missing_keys}")
            logger.info(f"pretrained decoder unexpect kyes | : {unexpect_keys}")

            # logger.info(f"load pretrained model for decoder")
            # decoder = checkpoint_utils.load_pretrained_component_from_model(
            #     component=decoder, checkpoint=args.load_pretrained_decoder_from
            # )

        return decoder

@register_model_architecture("my_transformer", "my_transformer_base")
def my_transformer_base_architecture(args):

    base_architecture(args)


@register_model_architecture("my_transformer", "my_transformer_big")
def my_big_architecture(args):

    transformer_vaswani_wmt_en_fr_big(args)