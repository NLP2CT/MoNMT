# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.checkpoint_utils import prune_state_dict
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
    BaseFairseqModel,
    FairseqMultiModel
)
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

from fairseq.models.momer import Momermodel, base_architecture

from .lm_decoder import LMTransformerDecoder
from .mlm_encoder import MLMTransformerEncoder
from .x_decoder import XTransformerDecoder

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("multilingual_momer")
class MultilingualMomermodel(BaseFairseqModel):
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

    def __init__(self, args, encoders, decoders, xdecoders):
        super().__init__()
        self.args = args
        assert encoders.keys() == decoders.keys() == xdecoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            assert isinstance(encoders[key], FairseqEncoder)
            assert isinstance(decoders[key], FairseqDecoder)
            assert isinstance(xdecoders[key], FairseqDecoder)
        self.models = nn.ModuleDict(
            {
                key: Momermodel(encoders[key], decoders[key], xdecoders[key])
                for key in self.keys
            }
        )
        self.lm_fusion = args.lm_fusion



    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        Momermodel.add_args(parser)
        parser.add_argument(
            "--share-encoder-embeddings",
            action="store_true",
            help="share encoder embeddings across languages",
        )
        parser.add_argument(
            "--share-decoder-embeddings",
            action="store_true",
            help="share decoder embeddings across languages",
        )
        parser.add_argument(
            "--share-xdecoder-embeddings",
            action="store_true",
            help="share decoder embeddings across languages",
        )
        parser.add_argument(
            "--share-encoders",
            action="store_true",
            help="share encoders across languages",
        )
        parser.add_argument(
            "--share-decoders",
            action="store_true",
            help="share decoders across languages",
        )
        parser.add_argument(
            "--share-xdecoders",
            action="store_true",
            help="share decoders across languages",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

        assert isinstance(task, MultilingualTranslationTask)
        # make sure all arguments are present in older models
        base_multilingual_architecture(args)

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_langs = [lang_pair.split("-")[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split("-")[1] for lang_pair in task.model_lang_pairs]

        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True
        if args.share_xdecoders:
            args.share_xdecoder_embeddings = True

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.share_all_embeddings:
            if args.encoder_embed_dim != args.decoder_embed_dim != args.xdecoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path != args.xdecoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=args.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=args.encoder_embed_path,
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            shared_xdecoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=src_langs,
                    embed_dim=args.encoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.encoder_embed_path,
                )
            if args.share_decoder_embeddings:
                shared_decoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=tgt_langs,
                    embed_dim=args.decoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.decoder_embed_path,
                )
            if args.share_xdecoder_embeddings:
                shared_xdecoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=tgt_langs,
                    embed_dim=args.xdecoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.xdecoder_embed_path,
                )
        # encoders/decoders for each language
        lang_encoders, lang_decoders, lang_xdecoders = {}, {}, {}

        def get_encoder(lang):
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                else:
                    encoder_embed_tokens = build_embedding(
                        task.dicts[lang],
                        args.encoder_embed_dim,
                        args.encoder_embed_path,
                    )
                lang_encoders[lang] = MLMTransformerDecoder(args, task.dicts[lang], encoder_embed_tokens)
            return lang_encoders[lang]

        def get_decoder(lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang],
                        args.decoder_embed_dim,
                        args.decoder_embed_path,
                    )
                lang_decoders[lang] = LMTransformerDecoder(args, task.dicts[lang], decoder_embed_tokens)
            return lang_decoders[lang]

        def get_xdecoder(lang):
            if lang not in lang_decoders:
                if shared_xdecoder_embed_tokens is not None:
                    xdecoder_embed_tokens = shared_xdecoder_embed_tokens
                else:
                    xdecoder_embed_tokens = build_embedding(
                        task.dicts[lang],
                        args.decoder_embed_dim,
                        args.decoder_embed_path,
                    )
                lang_xdecoders[lang] = XTransformerDecoder(args, task.dicts[lang], xdecoder_embed_tokens)
                # lang_xdecoders[lang] = cls._get_module_class(
                #     False, args, task.dicts[lang], xdecoder_embed_tokens, tgt_langs
                # )
            return lang_decoders[lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder, shared_xdecoder = None, None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])
        if args.share_xdecoders:
            shared_xdecoder = get_xdecoder(tgt_langs[0])

        encoders, decoders, xdecoders = OrderedDict(), OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = (
                shared_encoder if shared_encoder is not None else get_encoder(src)
            )
            decoders[lang_pair] = (
                shared_decoder if shared_decoder is not None else get_decoder(tgt)
            )
            xdecoders[lang_pair] = (
                shared_xdecoder if shared_xdecoder is not None else get_xdecoder(tgt)
            )

        return cls(args, encoders, decoders, xdecoders)
    
    def load_state_dict(self, state_dict, strict=True, model_cfg=None):
        state_dict_subset = state_dict.copy()
        for k, _ in state_dict.items():
            assert k.startswith("models.")
            lang_pair = k.split(".")[1]
            if lang_pair not in self.models:
                del state_dict_subset[k]
        if model_cfg is None and args is not None:
            logger.warn("using 'args' is deprecated, please update your code to use dataclass config")
            model_cfg = convert_namespace_to_omegaconf(args).model

        self.upgrade_state_dict(state_dict)
        new_state_dict = prune_state_dict(state_dict, model_cfg)
        return super().load_state_dict(new_state_dict, strict)


@register_model_architecture("multilingual_momer", "multilingual_momer")
def base_multilingual_architecture(args):
    base_architecture(args)
    args.share_encoder_embeddings = getattr(args, "share_encoder_embeddings", False)
    args.share_decoder_embeddings = getattr(args, "share_decoder_embeddings", False)
    args.share_xdecoder_embeddings = getattr(args, "share_xdecoder_embeddings", False)
    args.share_encoders = getattr(args, "share_encoders", False)
    args.share_decoders = getattr(args, "share_decoders", False)
    args.share_xdecoders = getattr(args, "share_xdecoders", False)

@register_model_architecture("multilingual_momer", "multilingual_share_momer")
def base_share_model_multilingual_architecture(args):
    args.share_encoder_embeddings = getattr(args, "share_encoder_embeddings", True)
    args.share_decoder_embeddings = getattr(args, "share_decoder_embeddings", True)
    args.share_xdecoder_embeddings = getattr(args, "share_xdecoder_embeddings", True)
    args.share_encoders = getattr(args, "share_encoders", True)
    args.share_decoders = getattr(args, "share_decoders", True)
    args.share_xdecoders = getattr(args, "share_xdecoders", True)
    base_multilingual_architecture(args)