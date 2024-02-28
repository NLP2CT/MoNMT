# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from .reencoder_layer import TransformerReEncoderLayer


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


class ReEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args):
        super().__init__()

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.reencoder_layerdrop = args.reencoder_layerdrop

        encoder_embed_dim = args.encoder_embed_dim
        reencoder_embed_dim = args.reencoder_embed_dim

        self.in_proj_layer=None
        self.out_proj_layer=None
        if encoder_embed_dim != reencoder_embed_dim:
            self.in_proj_layer = nn.Linear(encoder_embed_dim, reencoder_embed_dim)
            self.out_proj_layer = nn.Linear(reencoder_embed_dim, encoder_embed_dim)
        # self.max_source_positions = args.max_source_positions
        # self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)


        # if not args.adaptive_input and args.quant_noise_pq > 0:
        #     self.quant_noise = apply_quant_noise_(
        #         nn.Linear(embed_dim, embed_dim, bias=False),
        #         args.quant_noise_pq,
        #         args.quant_noise_pq_block_size,
        #     )
        # else:
        #     self.quant_noise = None

        if self.reencoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.reencoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_reencoder_layer(args) for i in range(args.reencoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.reencoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_reencoder_layer(self, args):
        layer = TransformerReEncoderLayer(args)
        if getattr(args, "checkpoint_activations", False):
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer

    def forward_prev_features(
        self, prev_features
    ):
        # embed tokens and positions
        if self.in_proj_layer is not None:
            return self.in_proj_layer(prev_features)
        return prev_features
    
    def forward_final_features(self, x):
        if self.out_proj_layer is not None:
            return self.out_proj_layer(x)
        return x

    def forward(
        self,
        src_tokens_prev_features,
        src_tokens_padding_masking,
        src_lang = None,
        tgt_lang = None,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        insert: bool = False
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x = self.forward_prev_features(src_tokens_prev_features) # T x B x C

        # compute padding mask
        encoder_padding_mask = src_tokens_padding_masking

        encoder_states = []

        # encoder layers
        for i, layer in enumerate(self.layers):
            # if i == 0:
            #     normalize_before = False
            # else:
            #     normalize_before = True
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None and not insert:
            x = self.layer_norm(x)

        x = self.forward_final_features(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `foward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }


    # def upgrade_state_dict_named(self, state_dict, name):
    #     """Upgrade a (possibly old) state dict for new versions of fairseq."""
    #     if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
    #         weights_key = "{}.embed_positions.weights".format(name)
    #         if weights_key in state_dict:
    #             print("deleting {0}".format(weights_key))
    #             del state_dict[weights_key]
    #         state_dict[
    #             "{}.embed_positions._float_tensor".format(name)
    #         ] = torch.FloatTensor(1)
    #     for i in range(self.num_layers):
    #         # update layer norms
    #         self.layers[i].upgrade_state_dict_named(
    #             state_dict, "{}.layers.{}".format(name, i)
    #         )

    #     version_key = "{}.version".format(name)
    #     if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
    #         # earlier checkpoints did not normalize after the stack of layers
    #         self.layer_norm = None
    #         self.normalize = False
    #         state_dict[version_key] = torch.Tensor([1])
    #     return state_dict

class Multilingual_Reencoder(nn.Module):

    def __init__(self, args, lang2fam={"all":"0"}):
        super().__init__()
        self.lang2fam = lang2fam
        self.fam2reencoder = dict()
        # if lang2fam is None:
        #     self.fam2reencoder['all'] = ReEncoder(args)
        # else:
        for k, v in lang2fam.items():
            if v not in self.fam2reencoder:
                self.fam2reencoder[v] = ReEncoder(args)

        if args.target_family_reencoder:
            self.target_family_reencoder = True
            self.source_family_reencoder = False
        elif args.source_family_reencoder:
            self.source_family_reencoder = True
            self.target_family_reencoder = False
        else:
            self.source_family_reencoder = False
            self.target_family_reencoder = False

        
    def forward(
        self,
        src_tokens_prev_features,
        src_tokens_padding_masking,
        src_lang = None,
        tgt_lang = None,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
    ):
        if self.source_family_reencoder:
            assert src_lang is not None and src_lang in self.lang2fam
            return self.fam2reencoder[self.lang2fam[src_lang]].forward(
                src_tokens_prev_features,
                src_tokens_padding_masking,
                src_lengths,
                return_all_hiddens
                )
        elif self.target_family_reencoder:
            assert tgt_lang is not None and tgt_lang in self.lang2fam
            return self.fam2reencoder[self.lang2fam[tgt_lang]].forward(
                src_tokens_prev_features,
                src_tokens_padding_masking,
                src_lengths,
                return_all_hiddens
                )
        else:
            return self.fam2reencoder[self.lang2fam["all"]].forward(
                src_tokens_prev_features,
                src_tokens_padding_masking,
                src_lengths,
                return_all_hiddens
                )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }